import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    """
    RDTRunner: 机器人扩散变换器运行器
    
    这个类实现了一个用于机器人控制的扩散模型，能够根据语言指令、图像和状态信息
    生成机器人动作序列。核心思想是将动作生成建模为一个去噪过程。
    
    继承关系：
    - nn.Module: PyTorch神经网络模块基类
    - CompatiblePyTorchModelHubMixin: 使模型能够与HuggingFace Hub兼容
    """
    
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        """
        初始化RDTRunner
        
        参数说明：
        - action_dim: 动作维度，表示机器人每个时间步的动作向量大小
        - pred_horizon: 预测时域，表示一次预测未来多少个时间步的动作
        - config: 配置字典，包含模型各组件的超参数
        - lang_token_dim: 语言token的特征维度
        - img_token_dim: 图像token的特征维度  
        - state_token_dim: 状态token的特征维度
        - max_lang_cond_len: 语言条件的最大长度
        - img_cond_len: 图像条件的长度
        - lang_pos_embed_config: 语言位置编码配置
        - img_pos_embed_config: 图像位置编码配置
        - dtype: 模型计算精度，默认使用bfloat16提高效率
        """
        super(RDTRunner, self).__init__()
        
        # 获取模型隐藏层维度
        hidden_size = config['rdt']['hidden_size']
        
        # ========= 创建核心扩散模型 =========
        self.model = RDT(
            output_dim=action_dim,                    # 输出动作维度
            horizon=pred_horizon,                     # 预测时域长度
            hidden_size=hidden_size,                  # 隐藏层维度
            depth=config['rdt']['depth'],             # Transformer层数
            num_heads=config['rdt']['num_heads'],     # 多头注意力头数
            max_lang_cond_len=max_lang_cond_len,      # 最大语言条件长度
            img_cond_len=img_cond_len,                # 图像条件长度
            lang_pos_embed_config=lang_pos_embed_config,  # 语言位置编码配置
            img_pos_embed_config=img_pos_embed_config,    # 图像位置编码配置
            dtype=dtype,                              # 数据类型
        )

        # ========= 创建条件输入适配器 =========
        # 这些适配器将不同模态的特征映射到统一的隐藏空间
        
        # 语言适配器：将语言特征映射到隐藏维度
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        
        # 图像适配器：将图像特征映射到隐藏维度
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        
        # 状态适配器：将状态特征映射到隐藏维度
        # 注意：输入特征维度是state_token_dim * 2，因为包含状态值和状态掩码
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    # 状态 + 状态掩码(指示器)
            out_features=hidden_size
        )
        
        # ========= 创建噪声调度器 =========
        noise_scheduler_config = config['noise_scheduler']
        
        # 训练时使用的DDPM调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],    # 训练时间步数
            beta_schedule=noise_scheduler_config['beta_schedule'],                # 噪声调度策略
            prediction_type=noise_scheduler_config['prediction_type'],            # 预测类型(epsilon/sample)
            clip_sample=noise_scheduler_config['clip_sample'],                    # 是否裁剪样本
        )
        
        # 推理时使用的DPM求解器调度器(更快的采样)
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        # 保存关键超参数
        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']        # 训练时间步
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps'] # 推理时间步
        self.prediction_type = noise_scheduler_config['prediction_type']                # 预测类型
        self.pred_horizon = pred_horizon                                               # 预测时域
        self.action_dim = action_dim                                                   # 动作维度

        # 打印模型参数量统计
        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """
        构建条件适配器
        
        根据配置类型构建不同的映射网络，将输入特征映射到目标维度
        
        参数：
        - projector_type: 投影器类型，支持'linear'或'mlp{n}x_gelu'格式
        - in_features: 输入特征维度
        - out_features: 输出特征维度
        
        返回：
        - projector: 构建好的投影器网络
        """
        projector = None
        
        if projector_type == 'linear':
            # 简单线性映射
            projector = nn.Linear(in_features, out_features)
        else:
            # 匹配MLP格式：mlp{depth}x_gelu，如mlp2x_gelu表示2层MLP
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))  # 提取MLP层数
                
                # 构建MLP网络
                modules = [nn.Linear(in_features, out_features)]  # 第一层
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))    # GELU激活函数
                    modules.append(nn.Linear(out_features, out_features))  # 后续层
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        """
        适配条件输入
        
        将不同模态的条件输入通过对应的适配器映射到统一的隐藏空间
        
        参数：
        - lang_tokens: 语言tokens，形状 (batch_size, lang_len, lang_token_dim)
        - img_tokens: 图像tokens，形状 (batch_size, img_len, img_token_dim)  
        - state_tokens: 状态tokens，形状 (batch_size, state_len, state_token_dim)
        
        返回：
        - adapted_lang: 适配后的语言特征，形状 (..., hidden_size)
        - adapted_img: 适配后的图像特征，形状 (..., hidden_size)
        - adapted_state: 适配后的状态特征，形状 (..., hidden_size)
        """
        adapted_lang = self.lang_adaptor(lang_tokens)     # 语言特征适配
        adapted_img = self.img_adaptor(img_tokens)        # 图像特征适配
        adapted_state = self.state_adaptor(state_tokens)  # 状态特征适配

        return adapted_lang, adapted_img, adapted_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        """
        条件采样生成动作序列
        
        使用扩散模型的逆过程，从噪声中逐步去噪生成动作序列
        
        参数：
        - lang_cond: 语言条件数据，形状 (batch_size, lang_len, hidden_size)
        - lang_attn_mask: 语言注意力掩码，形状 (batch_size, lang_len)，布尔张量
        - img_cond: 图像条件数据，形状 (batch_size, img_len, hidden_size)
        - state_traj: 状态轨迹，形状 (batch_size, 1, hidden_size)
        - action_mask: 动作掩码，形状 (batch_size, 1, action_dim)，浮点张量，指示有效动作维度
        - ctrl_freqs: 控制频率，形状 (batch_size,)，每个样本的控制频率
        
        返回：
        - 生成的动作序列，形状 (batch_size, horizon, action_dim)
        """
        device = state_traj.device
        dtype = state_traj.dtype
        
        # 初始化随机噪声作为起点, 形状 (batch_size, horizon, action_dim)
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        
        # 扩展动作掩码到整个预测时域
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # 设置推理时间步
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        # 逐步去噪过程
        for t in self.noise_scheduler_sample.timesteps:
            # 准备状态-动作轨迹
            # 将当前噪声动作与动作掩码拼接
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)  # 通过适配器处理
            
            # 将状态和动作轨迹拼接形成完整输入
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # 模型预测（去噪）
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),      # 时间步
                                    lang_cond, img_cond,             # 条件输入
                                    lang_mask=lang_attn_mask)        # 语言掩码
            
            # 执行去噪步骤：x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # 最终应用动作掩码，屏蔽无效动作维度
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= 训练相关方法 ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs
                    ) -> torch.Tensor:
        """
        计算训练损失
        
        实现扩散模型的前向过程：给干净动作添加噪声，然后训练模型预测噪声或原始动作
        
        参数：
        - lang_tokens: 语言tokens，形状 (batch_size, lang_len, lang_token_dim)
        - lang_attn_mask: 语言注意力掩码，形状 (batch_size, lang_len)，布尔张量  
        - img_tokens: 图像tokens，形状 (batch_size, img_len, img_token_dim)
        - state_tokens: 状态tokens，形状 (batch_size, 1, state_token_dim)
        - action_gt: 真实动作序列，形状 (batch_size, horizon, state_token_dim)
        - action_mask: 动作掩码，形状 (batch_size, 1, state_token_dim)，浮点张量
        - ctrl_freqs: 控制频率，形状 (batch_size,)
        
        返回：
        - loss_value: 标量损失值
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        # ========= 扩散前向过程 =========
        # 采样随机噪声
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        
        # 随机采样扩散时间步
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 根据时间步向干净动作添加噪声（前向扩散过程）
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # ========= 准备模型输入 =========
        # 拼接状态和噪声动作形成输入序列 (batch_size, horizon + 1, state_token_dim)
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1) # 
        
        # 扩展动作掩码并拼接到输入序列
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1) # (batch_size, horizon + 1, state_token_dim)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2) # (batch_size, horizon + 1, state_token_dim*2 )
        
        # 通过适配器将特征对齐到隐藏维度
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        
        # ========= 模型预测和损失计算 =========
        # 预测去噪结果
        pred = self.model(state_action_traj, ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        # 根据预测类型确定训练目标
        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise          # 预测噪声
        elif pred_type == 'sample':
            target = action_gt      # 预测原始动作
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 计算均方误差损失
        loss = F.mse_loss(pred, target)
        return loss
    
    # ========= 推理相关方法 ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs):
        """
        预测动作序列
        
        给定当前条件信息，预测未来的动作序列
        
        参数：
        - lang_tokens: 语言tokens，形状 (batch_size, lang_len, lang_token_dim)
        - lang_attn_mask: 语言注意力掩码，形状 (batch_size, lang_len)，布尔张量
        - img_tokens: 图像tokens，形状 (batch_size, img_len, img_token_dim)  
        - state_tokens: 状态tokens，形状 (batch_size, 1, state_token_dim)
        - action_mask: 动作掩码，形状 (batch_size, 1, action_dim)，浮点张量， action_dim  实际上是 state_token_dim?
        - ctrl_freqs: 控制频率，形状 (batch_size,)
        
        返回：
        - 预测的动作序列，形状 (batch_size, horizon, action_dim)
        """
        # 准备状态和条件
        # 将状态tokens与动作掩码拼接
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        
        # 通过适配器处理各种条件输入,这里的 state_traj (batch_size, 1, hidden_dim)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        # 运行条件采样生成动作
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播函数
        
        训练时调用compute_loss计算损失
        """
        return self.compute_loss(*args, **kwargs)