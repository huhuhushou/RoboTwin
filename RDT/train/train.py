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

import diffusers  # 扩散模型库
import torch
import torch.utils.checkpoint
import transformers  # HuggingFace transformers库
import yaml  # YAML配置文件解析
from accelerate import Accelerator  # 分布式训练加速器
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler  # 学习率调度器
from diffusers.utils import is_wandb_available  # wandb可用性检查
from huggingface_hub import create_repo, upload_folder  # HuggingFace Hub操作
from tqdm.auto import tqdm  # 进度条
from safetensors.torch import load_model  # 安全张量加载

# 自定义模块导入
from models.ema_model import EMAModel  # 指数移动平均模型
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower  # 视觉编码器
from models.multimodal_encoder.t5_encoder import T5Embedder  # 文本编码器
from models.rdt_runner import RDTRunner  # RDT模型运行器
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset  # 数据集相关
from train.sample import log_sample_res  # 采样结果记录

# 条件导入wandb
if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    """
    保存模型卡片到README.md文件
    
    Args:
        repo_id: 仓库ID
        base_model: 基础模型名称
        repo_folder: 仓库文件夹路径
    """
    # 定义模型卡片的YAML元数据
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
- diffusion
- rdt
---
    """
    # 定义模型卡片的描述内容
    model_card = f"""
# RDT - {repo_id}

This is a RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/).
"""
    # 将模型卡片写入README.md文件
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    """
    主训练函数
    
    Args:
        args: 训练参数对象
        logger: 日志记录器
    """
    # ================================ 配置文件读取 ================================
    # 读取数据集配置文件
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # 读取模型配置文件
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    # 设置输出目录为模型配置中的检查点路径
    args.output_dir = model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    # ================================ 加速器配置 ================================
    # 配置项目限制（检查点数量限制）
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    
    # 初始化Accelerator，支持分布式训练和混合精度
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        mixed_precision=args.mixed_precision,  # 混合精度训练
        log_with=args.report_to,  # 日志记录工具
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # 检查wandb可用性
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
    
    # 设置不同进程的日志级别
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
        # 创建输出目录
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # 如果需要推送到HuggingFace Hub，创建仓库
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # ================================ 数据类型配置 ================================
    # 根据混合精度设置确定权重数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ================================ 编码器初始化 ================================
    # 根据是否使用预计算语言嵌入来决定是否初始化文本编码器
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        # 初始化T5文本编码器
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # 初始化SigLIP视觉编码器
    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # ================================ RDT模型初始化 ================================
    # 从预训练检查点加载或从配置构建模型
    if args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path):
        logger.info("Constructing model from pretrained checkpoint.")
        rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        logger.info("Constructing model from provided config.")
        
        # 计算图像条件长度（历史图像数量 × 相机数量 × 每个图像的补丁数量）
        img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                        vision_encoder.num_patches)
        
        # 创建RDT模型实例
        rdt = RDTRunner(
            action_dim=config["common"]["state_dim"],  # 动作维度
            pred_horizon=config["common"]["action_chunk_size"],  # 预测时间范围
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],  # 语言token维度
            img_token_dim=config["model"]["img_token_dim"],  # 图像token维度
            state_token_dim=config["model"]["state_token_dim"],  # 状态token维度
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],  # 最大语言条件长度
            img_cond_len=img_cond_len,  # 图像条件长度
            # 图像位置嵌入配置
            img_pos_embed_config=[
                (
                    "image",
                    (
                        config["common"]["img_history_size"],
                        config["common"]["num_cameras"],
                        -vision_encoder.num_patches,  # 负数表示不使用初始位置嵌入
                    ),
                ),
            ],
            # 语言位置嵌入配置
            lang_pos_embed_config=[
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
        )

    # ================================ EMA模型初始化 ================================
    # 创建EMA（指数移动平均）模型用于稳定训练
    ema_rdt = copy.deepcopy(rdt)
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],  # 开始更新的步数
        inv_gamma=config["model"]["ema"]["inv_gamma"],  # 逆伽马参数
        power=config["model"]["ema"]["power"],  # 幂参数
        min_value=config["model"]["ema"]["min_value"],  # 最小值
        max_value=config["model"]["ema"]["max_value"],  # 最大值
    )

    # ================================ 模型保存钩子 ================================
    def save_model_hook(models, weights, output_dir):
        """自定义模型保存钩子，确保以HuggingFace格式保存"""
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
    # 在Ampere GPU上启用TF32以加速训练
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ================================ 学习率缩放 ================================
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                              accelerator.num_processes)

    # ================================ 优化器配置 ================================
    # 选择优化器（8位Adam或标准AdamW）
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 创建优化器
    params_to_optimize = rdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ================================ 数据集和数据加载器 ================================
    # 创建训练数据集（带数据增强）
    train_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,  # 图像增强
        cond_mask_prob=args.cond_mask_prob,  # 条件掩码概率
        cam_ext_mask_prob=args.cam_ext_mask_prob,  # 相机外参掩码概率
        state_noise_snr=args.state_noise_snr,  # 状态噪声信噪比
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    
    # 创建采样数据集（不带数据增强，用于验证）
    sample_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,  # 不使用图像增强
        cond_mask_prob=0,  # 不使用条件掩码
        cam_ext_mask_prob=-1,  # 不使用相机外参掩码
        state_noise_snr=None,  # 不添加状态噪声
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )

    # 数据整理器
    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

    # 创建训练数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,  # 将数据固定在内存中以加速GPU传输
        persistent_workers=True,  # 保持worker进程以减少启动开销
    )
    
    # 创建采样数据加载器
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
    # 计算训练步数
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ================================ Accelerator准备 ================================
    # 使用accelerator准备所有组件以支持分布式训练
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = (accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler))

    # 将EMA模型和编码器移动到设备
    ema_rdt.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # ================================ 重新计算训练步数 ================================
    # 数据加载器大小可能在prepare后发生变化，需要重新计算
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ================================ 跟踪器初始化 ================================
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "VLA",
            config=vars(args),
            init_kwargs={"wandb": {
                "name": f"RoboTwin_RDT_{args.CONFIG_NAME}",
            }},
        )

    # ================================ 训练信息打印 ================================
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # 初始化训练状态变量
    global_step = 0
    first_epoch = 0

    # ================================ 检查点加载 ================================
    # 从预训练检查点加载模型权重（如果提供的是文件路径）
    if (args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)):
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        rdt.module.load_state_dict(checkpoint["module"])

    # 从训练检查点恢复（继续训练）
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最新的检查点
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
                # 尝试加载完整的训练状态
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                # 如果失败，只加载模型权重
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(
                        args.output_dir,
                        path,
                        "pytorch_model",
                        "mp_rank_00_model_states.pt",
                    ))
                rdt.module.load_state_dict(checkpoint["module"])

            # 加载EMA模型
            load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            # 计算恢复位置
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

        rdt.train()  # 设置为训练模式

        # 如果从检查点恢复，设置正确的进度条位置
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        # ================================ 批次训练循环 ================================
        for batch in train_dataloader:
            # 梯度累积上下文
            with accelerator.accumulate(rdt):
                # 准备输入数据
                images = batch["images"].to(dtype=weight_dtype)  # 图像数据
                states = batch["states"].to(dtype=weight_dtype)  # 状态数据 (B, T, D_a)
                states = states[:, -1:, :]  # 只使用最后一个状态作为输入
                actions = batch["actions"].to(dtype=weight_dtype)  # 动作数据
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)  # 状态元素掩码
                ctrl_freqs = batch["ctrl_freqs"]  # 控制频率

                # ================================ 特征编码 ================================
                with torch.no_grad():
                    # 视觉特征编码
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    # 语言特征编码
                    lang_attn_mask = batch["lang_attn_mask"]
                    if args.precomp_lang_embed:
                        # 使用预计算的语言嵌入
                        text_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
                    else:
                        # 实时计算语言嵌入
                        text_embeds = text_encoder(
                            input_ids=batch["input_ids"], 
                            attention_mask=lang_attn_mask
                        )["last_hidden_state"].detach()

                # ================================ 前向传播和损失计算 ================================
                state_elem_mask = state_elem_mask.unsqueeze(1)
                loss = rdt(
                    lang_tokens=text_embeds,      # 语言token
                    lang_attn_mask=lang_attn_mask,  # 语言注意力掩码
                    img_tokens=image_embeds,      # 图像token
                    state_tokens=states,          # 状态token
                    action_gt=actions,            # 真实动作（ground truth）
                    action_mask=state_elem_mask,  # 动作掩码
                    ctrl_freqs=ctrl_freqs,        # 控制频率
                )

                # ================================ 反向传播和优化 ================================
                accelerator.backward(loss)  # 反向传播
                
                # 梯度同步时执行优化步骤
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)  # 梯度裁剪
                
                optimizer.step()  # 优化器步骤
                lr_scheduler.step()  # 学习率调度器步骤
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)  # 清零梯度

            # ================================ EMA模型更新 ================================
            ema_model.step(accelerator.unwrap_model(rdt))

            # ================================ 检查点保存和采样 ================================
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 定期保存检查点
                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)  # 保存训练状态
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdt, ema_save_path)  # 保存EMA模型
                    logger.info(f"Saved state to {save_path}")

                # 定期进行采样评估
                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        rdt,  # 使用当前模型（非EMA）进行采样
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
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)

            # 检查是否达到最大训练步数
            if global_step >= args.max_train_steps:
                break

    # ================================ 训练完成后的模型保存 ================================
    accelerator.wait_for_everyone()  # 等待所有进程完成
    
    if accelerator.is_main_process:
        # 保存最终模型
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"Saved Model to {args.output_dir}")

        # ================================ 推送到HuggingFace Hub ================================
        if args.push_to_hub:
            # 保存模型卡片
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            # 上传模型文件到Hub
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],  # 只上传指定类型的文件
            )

    # 结束训练
    accelerator.end_training()