# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn


#################################################################################
#               嵌入层：处理时间步和条件输入                                      #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    将标量时间步嵌入到向量表示中
    
    这个类是扩散模型的核心组件，负责将时间步t（一个简单的数字）
    转换为丰富的向量表示，让模型能够理解当前的噪声水平和去噪进度。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        # MLP网络：将正弦余弦编码转换为任务特定的表示
        # 两层全连接网络，中间使用SiLU激活函数
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # SiLU = x * sigmoid(x)，比ReLU更平滑
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        创建正弦余弦时间步嵌入
        
        这是一个数学上精妙的设计，使用不同频率的正弦余弦函数
        来编码时间步信息。这种编码有以下优点：
        1. 唯一性：每个时间步都有唯一的编码
        2. 平滑性：相邻时间步的编码相似
        3. 周期性：能够处理超出训练范围的时间步
        4. 多尺度：同时包含高频和低频信息
        
        参数:
        :param t: 形状为(N,)的一维张量，每个批次元素一个索引，可以是小数
        :param dim: 输出的维度
        :param max_period: 控制嵌入的最小频率
        :return: 形状为(N, D)的位置嵌入张量
        """
        # 参考：https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        
        # 计算一半的维度，因为我们要生成sin和cos两个分量
        half = dim // 2
        
        # 生成频率序列：从高频到低频的指数衰减序列
        # 这创建了一个从1.0到1/max_period的指数衰减序列，freqs = [1.0, 0.9, 0.8, ..., 0.0001]
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        
        # 计算每个时间步与每个频率的乘积
        # t[:, None] 形状: (N, 1), freqs[None] 形状: (1, half)
        # 广播后 args 形状: (N, half)
        args = t[:, None].float() * freqs[None]
        
        # 计算正弦和余弦值，然后拼接
        # cos(args) 和 sin(args) 都是 (N, half)
        # 拼接后得到 (N, dim) 的嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # 如果维度是奇数，添加零填充
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding.to(self.dtype)

    def forward(self, t):
        """前向传播：时间步 -> 频率嵌入 -> MLP变换 -> 最终表示"""
        # 第一步：将时间步转换为正弦余弦编码
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 第二步：通过MLP学习任务特定的时间表示
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                              交叉注意力层                                     #
#################################################################################
class CrossAttention(nn.Module):
    """
    交叉注意力层，支持Flash Attention优化
    
    这是RDT模型的核心创新之一。与自注意力不同，交叉注意力允许
    主序列（动作序列）与条件信息（语言或视觉）进行交互。
    
    核心思想：
    - Query来自主序列（我想做什么动作？）
    - Key和Value来自条件序列（语言指令或视觉信息）
    - 让动作序列"查询"条件信息以获得指导
    """
    fused_attn: Final[bool]  # 是否使用融合注意力优化

    def __init__(
        self,
        dim: int,                    # 嵌入维度
        num_heads: int = 8,          # 多头注意力的头数
        qkv_bias: bool = False,      # 是否在QKV投影中使用偏置
        qk_norm: bool = False,       # 是否对Q和K进行归一化
        attn_drop: float = 0,        # 注意力dropout率
        proj_drop: float = 0,        # 投影dropout率
        norm_layer: nn.Module = nn.LayerNorm,  # 归一化层类型
    ) -> None:
        super().__init__()
        # 确保维度能被头数整除
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子，防止softmax饱和
        self.fused_attn = use_fused_attn()  # 检查是否可以使用优化的融合注意力

        # 定义线性投影层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)      # Query投影：主序列 -> Q
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) # Key和Value投影：条件序列 -> K,V
        
        # 可选的Q和K归一化（提高训练稳定性）
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        交叉注意力前向传播
        
        参数:
        x: 主序列 (B, N, C) - 例如动作序列
        c: 条件序列 (B, L, C) - 例如语言指令或图像特征
        mask: 注意力掩码 (B, L) - True表示有效位置，False表示需要忽略
        
        返回:
        更新后的主序列 (B, N, C)
        """
        B, N, C = x.shape   # 批次大小，主序列长度，通道数
        _, L, _ = c.shape   # 条件序列长度
        
        # 1. 计算Query：从主序列生成查询
        # 重塑为多头格式：(B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 2. 计算Key和Value：从条件序列生成
        # kv形状：(B, L, 2, num_heads, head_dim) -> (2, B, num_heads, L, head_dim)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # 分离Key和Value
        
        # 3. 可选的Q和K归一化
        q, k = self.q_norm(q), self.k_norm(k)

        # 4. 处理注意力掩码
        # 将(B, L)的掩码扩展为(B, 1, N, L)以适配多头注意力
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)

        # 5. 计算注意力
        if self.fused_attn:
            # 使用PyTorch优化的融合注意力（更快更省内存）
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            # 手动实现注意力计算
            q = q * self.scale  # 缩放查询
            attn = q @ k.transpose(-2, -1)  # 计算注意力分数
            
            # 应用掩码：将无效位置设为负无穷，softmax后会变成0
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            
            attn = attn.softmax(dim=-1)  # 归一化注意力权重
            
            # 应用dropout
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            
            x = attn @ v  # 加权求和

        # 6. 重塑输出并投影
        # (B, num_heads, N, head_dim) -> (B, N, C)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)  # 输出投影
        
        # 7. 应用输出dropout
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        
        return x


#################################################################################
#                                RDT 模块                                       #
#################################################################################
class RDTBlock(nn.Module):
    """
    RDT块：结合自注意力和交叉注意力的条件处理单元
    
    这是RDT架构的核心构建块。每个RDTBlock包含三个主要组件：
    1. 自注意力：让序列内部的token相互交流
    2. 交叉注意力：让序列与外部条件信息交互
    3. 前馈网络：进行非线性变换和特征提取
    
    设计哲学：
    - 使用RmsNorm而不是LayerNorm（更稳定）
    - 每个组件都有残差连接（帮助梯度流动）
    - 先自注意力再交叉注意力（先内部整合，再外部查询）
    """

    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        
        # 第一个归一化层：用于自注意力
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        
        # 自注意力层：序列内部的token相互交流
        # 使用timm库的优化实现，支持QK归一化等高级特性
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,      # 使用偏置（有助于表达能力）
            qk_norm=True,       # Q和K归一化（提高稳定性）
            norm_layer=RmsNorm, # 使用RmsNorm而不是LayerNorm
            **block_kwargs
        )
        
        # 交叉注意力层：与条件信息交互
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            **block_kwargs
        )

        # 第二和第三个归一化层
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        
        # 前馈网络：使用GELU激活函数的MLP
        # 使用tanh近似的GELU（计算更高效）
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,  # 通常会设为4*hidden_size，这里保持不变
            act_layer=approx_gelu,
            drop=0
        )

    def forward(self, x, c, mask=None):
        """
        RDT块的前向传播
        
        参数:
        x: 主序列 (B, N, C)
        c: 条件序列 (B, L, C) 
        mask: 条件序列的掩码 (B, L)
        
        处理流程：
        1. 自注意力：让序列内部信息交流
        2. 交叉注意力：整合外部条件信息
        3. 前馈网络：非线性变换和特征提取
        每一步都有残差连接和归一化
        """
        
        # 第一步：自注意力处理
        # Pre-norm设计：先归一化再处理，然后添加残差
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)  # 自注意力：序列内部交流
        x = x + origin_x  # 残差连接

        # 第二步：交叉注意力处理  
        origin_x = x
        x = self.norm2(x)
        x = self.cross_attn(x, c, mask)  # 交叉注意力：与条件信息交互
        x = x + origin_x  # 残差连接

        # 第三步：前馈网络处理
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)   # 前馈网络：非线性变换
        x = x + origin_x  # 残差连接

        return x


class FinalLayer(nn.Module):
    """
    RDT的最终输出层
    
    将高维的内部表示转换为具体的动作空间输出。
    这是模型的"决策层"，负责将抽象的语义理解
    转换为机器人可以执行的具体动作。
    
    设计要点：
    - 使用RmsNorm确保输出稳定性
    - 使用MLP进行维度变换
    - 在初始化时权重设为零（扩散模型的重要技巧）
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        # 最终归一化层：确保输出稳定
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        
        # 最终的前馈网络：hidden_size -> out_channels
        # out_channels通常是动作空间的维度（如关节角度数量）
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,  # 中间层维度
            out_features=out_channels,    # 输出动作维度
            act_layer=approx_gelu,
            drop=0
        )

    def forward(self, x):
        """
        最终层前向传播
        
        输入: (B, T, hidden_size) - 内部表示
        输出: (B, T, out_channels) - 动作预测
        """
        x = self.norm_final(x)  # 归一化
        x = self.ffn_final(x)   # 维度变换
        return x


#################################################################################
#                      正弦/余弦位置编码函数                                    #
#################################################################################

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从一维位置网格生成正弦余弦位置嵌入
    
    这是位置编码的基础函数，使用不同频率的正弦余弦函数
    来编码位置信息。这种编码方式最早在Transformer论文中提出。
    
    参数:
    embed_dim: 每个位置的输出维度
    pos: 要编码的位置列表，大小为(M,)
    返回: (M, D)的位置嵌入
    """
    # 确保嵌入维度是偶数（因为要分配给sin和cos）
    assert embed_dim % 2 == 0
    
    # 创建频率序列：omega = 1 / (10000^(2i/d))
    # 这创建了从高频到低频的序列
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,) 频率序列

    # 确保pos是numpy数组
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,) 展平位置数组
    
    # 计算位置与频率的外积：pos * omega
    # einsum('m,d->md') 相当于 pos[:, None] * omega[None, :]
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    # 计算正弦和余弦分量
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # 拼接sin和cos分量
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    从多维位置网格生成正弦余弦位置嵌入
    
    这个函数处理多维位置编码，比如2D图像或3D体素。
    对于每个维度，它分配一部分嵌入维度来编码该维度的位置信息。
    
    参数:
    embed_dim: 每个位置的输出维度
    grid_sizes: 每个维度的网格大小 (K,)
    返回: (grid_sizes[0], ..., grid_sizes[K-1], D)的位置嵌入
    """
    num_sizes = len(grid_sizes)
    # 对于大小为1的网格，不需要添加位置嵌入
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    
    # 初始化嵌入数组
    emb = np.zeros(grid_sizes + (embed_dim, ))
    
    # 为每个网格维度均匀分配嵌入维度
    dim_for_each_grid = embed_dim // num_valid_sizes
    # 确保是偶数（因为sin/cos需要成对）
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        # 跳过大小为1或更小的维度
        if grid_size <= 1:
            continue
            
        # 为当前维度生成位置编码
        pos = np.arange(grid_size)
        
        # 创建正确的广播形状
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1  # 当前维度设为-1（自动推断）
        
        # 生成1D位置嵌入并重塑为正确形状，然后添加到总嵌入中
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        
        valid_size_idx += 1
    
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, embed_modality=True):
    """
    为多模态条件生成位置嵌入
    
    这是RDT模型的关键函数，它处理不同模态（语言、图像、动作等）
    的位置编码需求。每个模态可能有不同的结构特性：
    - 语言：一维序列
    - 图像：二维空间网格  
    - 动作：一维时间序列
    
    参数:
    embed_dim: 嵌入维度
    mm_cond_lens: 有序字典，包含(模态名称, 模态token长度)对
                  对于"image"模态，值可以是多维元组
                  如果长度<0，表示该模态或网格没有位置嵌入
    embed_modality: 是否嵌入模态信息。默认为True
    
    返回:
    所有模态的连接位置嵌入
    """
    num_modalities = len(mm_cond_lens)
    
    # 初始化模态位置嵌入
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    
    if embed_modality:
        # 获取各种模态的嵌入（放在前半部分）
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, 
            torch.arange(num_modalities)
        )
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # 后半部分用于位置嵌入
        pos_embed_dim = embed_dim // 2
    else:
        # 整个嵌入都用于位置嵌入
        pos_embed_dim = embed_dim

    # 获取每个模态内部位置的嵌入
    c_pos_emb = np.zeros((0, embed_dim))
    
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            # 处理图像模态：可能是多维的（如2D图像）
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            
            # 生成多维位置嵌入
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes
            )
            
            # 创建完整的条件位置嵌入
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim, ))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            # 处理其他模态：一维序列（如时间、频率、动作、语言）
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim,
                torch.arange(cond_len if cond_len > 0 else 1)
            )
            
            # 创建条件位置嵌入
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        
        # 添加模态特定的嵌入
        cond_pos_embed += modality_pos_embed[idx]
        
        # 连接到总的位置嵌入
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)

    return c_pos_emb