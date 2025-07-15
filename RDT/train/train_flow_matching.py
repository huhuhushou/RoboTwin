#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# ================================ 导入依赖库 ================================
import copy
import logging
import math
import os
from pathlib import Path

import diffusers  
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers  
import yaml  
from accelerate import Accelerator  
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler  
from diffusers.utils import is_wandb_available  
from huggingface_hub import create_repo, upload_folder  
from tqdm.auto import tqdm  
from safetensors.torch import load_model  

# 自定义模块导入
from models.ema_model import EMAModel  
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower  
from models.multimodal_encoder.t5_encoder import T5Embedder  
from models.rdt_runner import RDTRunner  
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset  
from train.sample import log_sample_res  

# 条件导入wandb
if is_wandb_available():
    import wandb


# ================================ 流匹配相关类 ================================
class FlowMatchingScheduler:
    """流匹配调度器，用于时间采样和流的计算"""
    
    def __init__(self, sigma_min=0.001):
        self.sigma_min = sigma_min
    
    def sample_time(self, batch_size, device):
        """均匀采样时间 t ∈ [0, 1]"""
        return torch.rand(batch_size, device=device)
    
    def compute_flow(self, x0, x1, t):
        """计算从x0到x1的线性插值流
        
        Args:
            x0: 起始点（噪声）
            x1: 目标点（数据）
            t: 时间 ∈ [0, 1]
        
        Returns:
            xt: 插值点
            ut: 目标速度场
        """
        # 扩展时间维度以匹配数据维度
        t_expanded = t.view(-1, *([1] * (x1.dim() - 1)))
        
        # 线性插值: xt = (1-t) * x0 + t * x1
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # 目标速度场: ut = x1 - x0
        ut = x1 - x0
        
        return xt, ut


class FlowMatchingLoss:
    """流匹配损失函数"""
    
    def __init__(self):
        self.scheduler = FlowMatchingScheduler()
    
    def compute_loss(self, model, x1, cond_kwargs):
        """计算流匹配损失
        
        Args:
            model: 神经网络模型，预测速度场
            x1: 目标数据（如动作序列）
            cond_kwargs: 条件信息（语言、图像、状态等）
        
        Returns:
            loss: 流匹配损失
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 采样时间
        t = self.scheduler.sample_time(batch_size, device)
        
        # 采样起始噪声
        x0 = torch.randn_like(x1)
        
        # 计算插值和目标速度场
        xt, ut_target = self.scheduler.compute_flow(x0, x1, t)
        
        # 模型预测速度场
        ut_pred = model(
            x=xt,
            t=t,
            **cond_kwargs
        )
        
        # 计算MSE损失
        loss = F.mse_loss(ut_pred, ut_target)
        
        return loss


# ================================ 修改的RDTRunner类 ================================
class RDTRunnerFlowMatching(RDTRunner):
    """支持流匹配的RDT运行器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化流匹配损失
        self.flow_matching_loss = FlowMatchingLoss()
    
    def forward(self, lang_tokens, lang_attn_mask, img_tokens, 
                state_tokens, action_gt, action_mask, ctrl_freqs):
        """前向传播，使用流匹配损失
        
        Args:
            lang_tokens: 语言token
            lang_attn_mask: 语言注意力掩码
            img_tokens: 图像token
            state_tokens: 状态token
            action_gt: 真实动作
            action_mask: 动作掩码
            ctrl_freqs: 控制频率
        
        Returns:
            loss: 流匹配损失
        """
        # 准备条件信息
        cond_kwargs = {
            "lang_tokens": lang_tokens,
            "lang_attn_mask": lang_attn_mask,
            "img_tokens": img_tokens,
            "state_tokens": state_tokens,
            "action_mask": action_mask,
            "ctrl_freqs": ctrl_freqs,
        }
        
        # 计算流匹配损失
        loss = self.flow_matching_loss.compute_loss(
            model=self.model,  # 假设self.model是底层的transformer模型
            x1=action_gt,
            cond_kwargs=cond_kwargs
        )
        
        return loss
    
    def sample(self, cond_kwargs, num_inference_steps=50):
        """使用ODE求解器进行采样
        
        Args:
            cond_kwargs: 条件信息
            num_inference_steps: ODE求解步数
        
        Returns:
            samples: 生成的动作序列
        """
        batch_size = cond_kwargs["lang_tokens"].shape[0]
        device = cond_kwargs["lang_tokens"].device
        
        # 从噪声开始
        x = torch.randn(batch_size, self.pred_horizon, self.action_dim, device=device)
        
        # 时间步
        dt = 1.0 / num_inference_steps
        
        # ODE求解（简单的欧拉方法）
        for i in range(num_inference_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            # 预测速度场
            v = self.model(
                x=x,
                t=t,
                **cond_kwargs
            )
            
            # 更新x
            x = x + v * dt
        
        return x


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    """保存模型卡片到README.md文件"""
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- flow-matching
- rdt
---
    """
    model_card = f"""
# RDT Flow Matching - {repo_id}

This is a RDT model with Flow Matching derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/) with Flow Matching instead of DDPM.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    """主训练函数（流匹配版本）"""
    
    # ================================ 配置文件读取 ================================
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    args.output_dir = model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    # ================================ 加速器配置 ================================
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # ================================ 日志配置 ================================
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # ================================ 随机种子设置 ================================
    if args.seed is not None:
        set_seed(args.seed)

    # ================================ 仓库创建 ================================
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # ================================ 数据类型配置 ================================
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ================================ 编码器初始化 ================================
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # ================================ RDT模型初始化（流匹配版本）================================
    if args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path):
        logger.info("Constructing model from pretrained checkpoint.")
        # 注意：这里需要使用FlowMatching版本的RDTRunner
        rdt = RDTRunnerFlowMatching.from_pretrained(args.pretrained_model_name_or_path)
    else:
        logger.info("Constructing model from provided config.")
        
        img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                        vision_encoder.num_patches)
        
        rdt = RDTRunnerFlowMatching(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        config["common"]["img_history_size"],
                        config["common"]["num_cameras"],
                        -vision_encoder.num_patches,
                    ),
                ),
            ],
            lang_pos_embed_config=[
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
        )

    # ================================ EMA模型初始化 ================================
    ema_rdt = copy.deepcopy(rdt)
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],
        inv_gamma=config["model"]["ema"]["inv_gamma"],
        power=config["model"]["ema"]["power"],
        min_value=config["model"]["ema"]["min_value"],
        max_value=config["model"]["ema"]["max_value"],
    )

    # ================================ 模型保存钩子 ================================
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    # ================================ 梯度检查点 ================================
    if args.gradient_checkpointing:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # ================================ TF32优化 ================================
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ================================ 学习率缩放 ================================
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                              accelerator.num_processes)

    # ================================ 优化器配置 ================================
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = rdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ================================ 数据集和数据加载器 ================================
    train_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    
    sample_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )

    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # ================================ 学习率调度器 ================================
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ================================ Accelerator准备 ================================
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = (accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler))

    ema_rdt.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # ================================ 重新计算训练步数 ================================
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ================================ 跟踪器初始化 ================================
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "VLA_FlowMatching",
            config=vars(args),
            init_kwargs={"wandb": {
                "name": f"RoboTwin_RDT_FlowMatching_{args.CONFIG_NAME}",
            }},
        )

    # ================================ 训练信息打印 ================================
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running training with Flow Matching *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # ================================ 检查点加载 ================================
    if (args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)):
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        rdt.module.load_state_dict(checkpoint["module"])

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(
                        args.output_dir,
                        path,
                        "pytorch_model",
                        "mp_rank_00_model_states.pt",
                    ))
                rdt.module.load_state_dict(checkpoint["module"])

            load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # ================================ 进度条初始化 ================================
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # ================================ 主训练循环 ================================
    loss_for_log = {}
    for epoch in range(first_epoch, args.num_train_epochs):

        rdt.train()

        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        # ================================ 批次训练循环 ================================
        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                # 准备输入数据
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)
                states = states[:, -1:, :]  # 只使用最后一个状态
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]

                # ================================ 特征编码 ================================
                with torch.no_grad():
                    # 视觉特征编码
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    # 语言特征编码
                    lang_attn_mask = batch["lang_attn_mask"]
                    if args.precomp_lang_embed:
                        text_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
                    else:
                        text_embeds = text_encoder(
                            input_ids=batch["input_ids"], 
                            attention_mask=lang_attn_mask
                        )["last_hidden_state"].detach()

                # ================================ 前向传播和损失计算（流匹配）================================
                state_elem_mask = state_elem_mask.unsqueeze(1)
                loss = rdt(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                )

                # ================================ 反向传播和优化 ================================
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # ================================ EMA模型更新 ================================
            ema_model.step(accelerator.unwrap_model(rdt))

            # ================================ 检查点保存和采样 ================================
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdt, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    # 注意：这里需要修改log_sample_res函数以支持流匹配采样
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        rdt,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            # ================================ 日志记录 ================================
            logs = {"flow_matching_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # ================================ 训练完成后的模型保存 ================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"Saved Flow Matching Model to {args.output_dir}")

        # ================================ 推送到HuggingFace Hub ================================
        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training (Flow Matching)",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()