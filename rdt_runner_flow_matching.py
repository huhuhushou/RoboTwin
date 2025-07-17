import re
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
import numpy as np

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class FlowMatchingScheduler:
    """
    流匹配调度器
    
    实现条件流匹配(Conditional Flow Matching)，通过学习向量场来建模
    从噪声分布到数据分布的连续变换路径。
    """
    
    def __init__(self, sigma_min=1e-4, sigma_max=1.0, prediction_type="velocity"):
        """
        初始化流匹配调度器
        
        参数：
        - sigma_min: 最小噪声水平
        - sigma_max: 最大噪声水平  
        - prediction_type: 预测类型，目前支持"velocity"
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        
    def sample_time(self, batch_size, device):
        """
        采样时间步 t ~ Uniform(0, 1)
        """
        return torch.rand(batch_size, device=device)
    
    def marginal_prob(self, x, t):
        """
        计算边际概率的均值和标准差
        
        对于线性插值路径: x_t = (1-t)*x_0 + t*x_1 + sigma*epsilon
        其中 x_0 是噪声，x_1 是数据
        """
        # 重新参数化：x_t = t*x_1 + (1-t)*x_0 + sigma*epsilon
        # 这里 x_0 ~ N(0,I), x_1 是数据
        mean_coeff = t.view(-1, 1, 1)  # 数据的系数
        noise_coeff = (1 - t).view(-1, 1, 1)  # 噪声的系数
        
        # 方差调度：sigma(t) = sigma_min + (sigma_max - sigma_min) * (1-t)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        sigma = sigma.view(-1, 1, 1)
        
        return mean_coeff, noise_coeff, sigma
    
    def forward_sample(self, x1, t):
        """
        前向采样过程：给定数据 x1 和时间 t，采样 x_t
        
        参数：
        - x1: 真实数据，形状 (batch_size, ...)
        - t: 时间步，形状 (batch_size,)
        
        返回：
        - x_t: 噪声数据，形状与x1相同
        - x_0: 采样的噪声，形状与x1相同
        """
        # 采样基础噪声
        x_0 = torch.randn_like(x1)
        
        # 获取插值系数
        mean_coeff, noise_coeff, sigma = self.marginal_prob(x1, t)
        
        # 采样额外噪声
        epsilon = torch.randn_like(x1)
        
        # 线性插值 + 噪声
        x_t = mean_coeff * x1 + noise_coeff * x_0 + sigma * epsilon
        
        return x_t, x_0
    
    def compute_target_velocity(self, x1, x_0, t):
        """
        计算目标向量场 v_t
        
        对于线性插值路径，向量场为：
        v_t = d/dt[x_t] = x_1 - x_0
        
        参数：
        - x1: 真实数据
        - x_0: 噪声
        - t: 时间步
        
        返回：
        - 目标向量场
        """
        return x1 - x_0


class ODESolver:
    """
    ODE求解器，用于流匹配的推理过程
    """
    
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        
    def solve(self, x_0, lang_cond, img_cond, state_traj, action_mask, ctrl_freqs,
              t_span=(0.0, 1.0), num_steps=50, method='dopri5'):
        """
        求解ODE生成样本
        
        参数：
        - x_0: 初始噪声
        - lang_cond, img_cond, state_traj: 条件输入
        - action_mask: 动作掩码
        - ctrl_freqs: 控制频率
        - t_span: 时间范围
        - num_steps: 积分步数
        - method: 求解方法
        """
        device = x_0.device
        
        def ode_func(t, x):
            """ODE函数：dx/dt = v_t(x)"""
            # 转换为tensor
            t_tensor = torch.tensor(t, device=device, dtype=x_0.dtype)
            x_tensor = torch.tensor(x, device=device, dtype=x_0.dtype)
            x_tensor = x_tensor.view(x_0.shape)
            
            # 扩展时间步到batch维度
            t_batch = t_tensor.unsqueeze(0).repeat(x_0.shape[0])
            
            # 模型预测向量场
            with torch.no_grad():
                velocity = self.model(
                    state_traj, ctrl_freqs, t_batch,
                    lang_cond, img_cond, lang_mask=None
                )
            
            return velocity.detach().cpu().numpy().flatten()
        
        # 使用scipy求解ODE
        x_0_flat = x_0.detach().cpu().numpy().flatten()
        
        # 求解ODE
        sol = solve_ivp(
            ode_func, t_span, x_0_flat,
            t_eval=np.linspace(t_span[0], t_span[1], num_steps),
            method=method, rtol=1e-5, atol=1e-8
        )
        
        # 返回最终结果
        x_final = torch.tensor(sol.y[:, -1], device=device, dtype=x_0.dtype)
        x_final = x_final.view(x_0.shape)
        
        return x_final


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    """
    RDTRunner: 机器人扩散变换器运行器 (流匹配版本)
    
    这个类实现了一个用于机器人控制的流匹配模型，能够根据语言指令、图像和状态信息
    生成机器人动作序列。核心思想是学习一个向量场来建模从噪声到动作的连续变换。
    """
    
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        """
        初始化RDTRunner
        """
        super(RDTRunner, self).__init__()
        
        # 获取模型隐藏层维度
        hidden_size = config['rdt']['hidden_size']
        
        # ========= 创建核心流匹配模型 =========
        self.model = RDT(
            output_dim=action_dim,                    # 输出动作维度
            horizon=pred_horizon,                     # 预测时域长度
            hidden_size=hidden_size,                  # 隐藏层维度
            depth=config['rdt']['depth'],             # Transformer层数
            num_heads=config['rdt']['num_heads'],     # 多头注意力头数
            max_lang_cond_len=max_lang_cond_len,      # 最大语言条件长度
            img_cond_len=img_cond_len,                # 图像条件长度
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # ========= 创建条件输入适配器 =========
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,
            out_features=hidden_size
        )
        
        # ========= 创建流匹配调度器 =========
        flow_config = config.get('flow_matching', {})
        
        self.flow_scheduler = FlowMatchingScheduler(
            sigma_min=flow_config.get('sigma_min', 1e-4),
            sigma_max=flow_config.get('sigma_max', 1.0),
            prediction_type=flow_config.get('prediction_type', 'velocity')
        )
        
        # 创建ODE求解器
        self.ode_solver = ODESolver(self.model, self.flow_scheduler)
        
        # 保存关键超参数
        self.num_inference_steps = flow_config.get('num_inference_steps', 50)
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # 打印模型参数量统计
        print("Flow Matching params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """构建条件适配器"""
        projector = None
        
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        """适配条件输入"""
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)

        return adapted_lang, adapted_img, adapted_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        """
        条件采样生成动作序列 (流匹配版本)
        
        使用ODE求解器从噪声生成动作序列
        """
        device = state_traj.device
        dtype = state_traj.dtype
        
        # 初始化随机噪声作为起点
        x_0 = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device
        )
        
        # 扩展动作掩码
        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        
        # 准备条件输入
        def create_state_action_input(x_t):
            """创建状态-动作输入"""
            action_traj = torch.cat([x_t, action_mask_expanded], dim=2)
            action_traj = self.state_adaptor(action_traj)
            return torch.cat([state_traj, action_traj], dim=1)
        
        # 使用简化的欧拉方法进行ODE求解
        dt = 1.0 / self.num_inference_steps
        x_t = x_0
        
        for i in range(self.num_inference_steps):
            t = i * dt
            t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=dtype)
            
            # 准备输入
            state_action_input = create_state_action_input(x_t)
            
            # 预测向量场
            with torch.no_grad():
                velocity = self.model(
                    state_action_input, ctrl_freqs, t_tensor,
                    lang_cond, img_cond, lang_mask=lang_attn_mask
                )
            
            # 欧拉步骤: x_{t+dt} = x_t + dt * v_t(x_t)
            x_t = x_t + dt * velocity
            x_t = x_t.to(dtype)
        
        # 应用动作掩码
        x_t = x_t * action_mask_expanded
        
        return x_t
    
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs
                    ) -> torch.Tensor:
        """
        计算流匹配训练损失
        
        训练模型预测向量场 v_t，使得沿着向量场积分能从噪声到达数据
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        
        # ========= 流匹配前向过程 =========
        # 采样时间步 t ~ Uniform(0, 1)
        t = self.flow_scheduler.sample_time(batch_size, device)
        
        # 前向采样得到 x_t 和对应的噪声 x_0
        x_t, x_0 = self.flow_scheduler.forward_sample(action_gt, t)
        
        # 计算目标向量场
        target_velocity = self.flow_scheduler.compute_target_velocity(action_gt, x_0, t)
        
        # ========= 准备模型输入 =========
        # 拼接状态和噪声动作
        state_action_traj = torch.cat([state_tokens, x_t], dim=1)
        
        # 扩展动作掩码并拼接
        action_mask_expanded = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask_expanded], dim=2)
        
        # 通过适配器处理条件
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj
        )
        
        # ========= 模型预测和损失计算 =========
        # 预测向量场
        pred_velocity = self.model(
            state_action_traj, ctrl_freqs, t,
            lang_cond, img_cond, lang_mask=lang_attn_mask
        )
        
        # 计算向量场匹配损失
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss
    
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs):
        """
        预测动作序列 (流匹配版本)
        """
        # 准备条件输入
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        
        # 通过适配器处理条件
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens
        )
        
        # 使用流匹配采样生成动作
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs
        )
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播函数"""
        return self.compute_loss(*args, **kwargs)