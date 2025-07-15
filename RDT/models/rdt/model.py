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

from collections import OrderedDict
import torch
import torch.nn as nn
from pathlib import Path
import sys, os

# 获取当前工作空间路径，用于导入自定义模块
current_file = Path(__file__)
sys.path.append(current_file.parent.parent)

# 导入RDT模型所需的核心组件
from rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid,
                        get_multimodal_cond_pos_embed)


class RDT(nn.Module):
    """
    机器人扩散变换器(Robotics Diffusion Transformers)类
    
    这是一个专门为机器人任务设计的扩散模型，它结合了：
    1. 扩散过程：通过逐步去噪来生成动作序列
    2. 变换器架构：利用自注意力机制处理序列数据
    3. 多模态条件：同时处理语言指令和视觉信息
    
    核心思想：给定当前状态、语言指令和图像信息，预测未来的动作序列
    """

    def __init__(self,
                 output_dim=128,              # 输出维度：每个动作向量的维度
                 horizon=32,                  # 时间范围：预测未来多少步的动作
                 hidden_size=1152,            # 隐藏层维度：变换器的内部表示维度
                 depth=28,                    # 网络深度：变换器层的数量
                 num_heads=16,                # 注意力头数：多头注意力机制的头数
                 max_lang_cond_len=1024,      # 最大语言条件长度：语言指令的最大token数
                 img_cond_len=4096,           # 图像条件长度：图像特征的token数量
                 lang_pos_embed_config=None,  # 语言位置编码配置
                 img_pos_embed_config=None,   # 图像位置编码配置
                 dtype=torch.bfloat16):       # 数据类型：使用bfloat16提高训练效率
        super().__init__()
        
        # 保存关键参数供后续使用
        self.horizon = horizon                          # 动作序列的长度
        self.hidden_size = hidden_size                  # 所有token的统一表示维度
        self.max_lang_cond_len = max_lang_cond_len     # 语言条件的最大长度
        self.img_cond_len = img_cond_len               # 图像条件的固定长度
        self.dtype = dtype                             # 模型参数的数据类型
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        # 时间步嵌入器：将扩散过程的时间步转换为向量表示
        # 在扩散模型中，不同的时间步对应不同的噪声水平
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 控制频率嵌入器：将机器人的控制频率编码为向量
        # 不同任务可能需要不同的控制频率（如抓取vs导航）
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # 位置编码参数（可学习的）
        # 主序列位置编码：[时间步, 控制频率, 状态, 动作序列]
        # horizon + 3 = 动作序列长度 + 时间步 + 控制频率 + 当前状态
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3, hidden_size))
        
        # 语言条件位置编码：为语言指令的每个token提供位置信息
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        
        # 图像条件位置编码：为图像特征的每个空间位置提供位置信息
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        # 变换器块：核心的自注意力和前馈网络层
        # depth层的RDTBlock，每层都包含多头自注意力和前馈网络
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        
        # 最终输出层：将隐藏表示转换为最终的动作预测
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        # 初始化所有权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        权重初始化方法
        
        良好的权重初始化对于深度学习模型的训练稳定性至关重要。
        这里使用了多种初始化策略来确保模型训练的稳定性。
        """
        
        # 基础初始化函数：对线性层使用Xavier均匀初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # Xavier初始化有助于保持前向和反向传播时的梯度尺度
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # 偏置初始化为0是标准做法
                    nn.init.constant_(module.bias, 0)

        # 应用基础初始化到所有模块
        self.apply(_basic_init)

        # 初始化主序列的位置编码
        # 使用正弦-余弦位置编码，这是Transformer中的经典做法
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),      # 扩散时间步：1个token
                ('ctrl_freq', 1),     # 控制频率：1个token  
                ('state', 1),         # 当前状态：1个token
                ('action', self.horizon),  # 动作序列：horizon个token
            ])
        )
        # 将numpy数组转换为PyTorch参数
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # 初始化语言条件的位置编码
        if self.lang_pos_embed_config is None:
            # 默认使用简单的1D正弦-余弦位置编码
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size,
                torch.arange(self.max_lang_cond_len)
            )
        else:
            # 使用自定义的多模态位置编码配置
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        # 初始化图像条件的位置编码
        if self.img_pos_embed_config is None:
            # 默认使用简单的1D正弦-余弦位置编码
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, 
                torch.arange(self.img_cond_len)
            )
        else:
            # 使用自定义的多模态位置编码配置
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # 初始化时间步和控制频率嵌入器的MLP权重
        # 使用较小的标准差进行正态分布初始化，有助于训练稳定性
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # 零初始化最终层的输出权重和偏置
        # 这是扩散模型中的重要技巧，确保模型初始时输出接近零
        # 这样有助于扩散过程的稳定性
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

        # 将所有参数转换到指定的数据类型
        # bfloat16可以在保持精度的同时提高训练速度和减少显存使用
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        RDT模型的前向传播
        
        这个方法实现了完整的推理过程：
        1. 处理输入的状态和动作序列
        2. 嵌入时间步和控制频率信息
        3. 添加位置编码
        4. 通过变换器块处理序列
        5. 输出最终的动作预测
        
        参数说明：
        x: (B, T, D) 状态+动作token序列，T = horizon + 1
           其中包含当前状态和要预测的动作序列
           维度D假设与hidden_size相同
        freq: (B,) 标量，表示控制频率
              不同的机器人任务可能需要不同的控制频率
        t: (B,) 或 (1,) 扩散时间步
           在扩散过程中，t表示当前的噪声水平
        lang_c: (B, L_lang, D) 或 None，语言条件tokens（可变长度）
                包含任务的语言描述，如"pick up the red cup"
        img_c: (B, L_img, D) 或 None，图像条件tokens（固定长度）
               包含视觉观察信息，如相机图像的特征
        lang_mask: (B, L_lang) 或 None，语言条件掩码（True表示有效）
                   用于处理变长的语言序列
        img_mask: (B, L_img) 或 None，图像条件掩码（True表示有效）
                  用于处理可能缺失的图像信息
        
        返回：
        x: (B, horizon, output_dim) 预测的动作序列
        """
        
        # 步骤1：处理时间步嵌入
        # 将扩散时间步转换为向量表示并增加维度
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) 或 (1, 1, D)
        
        # 步骤2：处理控制频率嵌入
        # 将控制频率转换为向量表示
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 步骤3：处理时间步的广播
        # 如果时间步是标量（所有样本共享），则扩展到批次大小
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        # 步骤4：构建完整的输入序列
        # 将时间步、控制频率和原始序列连接起来
        # 最终序列：[时间步, 控制频率, 当前状态, 动作序列]
        x = torch.cat([t, freq, x], dim=1)  # (B, T+2 = horizon + 3, D)

        # 步骤5：添加位置编码
        # 主序列位置编码：告诉模型每个token在序列中的位置
        x = x + self.x_pos_embed
        
        # 语言条件位置编码：处理变长语言序列
        # 只取实际使用的长度部分
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        
        # 图像条件位置编码：处理固定长度图像序列
        img_c = img_c + self.img_cond_pos_embed

        # 步骤6：通过变换器块进行处理
        # 交替使用语言和图像条件进行注意力计算
        # 这种设计允许模型在不同层关注不同的模态信息
        conds = [lang_c, img_c]      # 条件序列列表
        masks = [lang_mask, img_mask]  # 对应的掩码列表
        
        for i, block in enumerate(self.blocks):
            # 交替选择条件：偶数层使用语言条件，奇数层使用图像条件
            c, mask = conds[i % 2], masks[i % 2]
            # 通过RDT块处理，包含自注意力和交叉注意力
            x = block(x, c, mask)  # (B, T+2, D)
        
        # 步骤7：生成最终输出
        # 通过最终层将隐藏表示转换为动作空间
        x = self.final_layer(x)  # (B, T+2, output_dim)

        # 步骤8：提取动作预测
        # 只保留动作tokens，去除时间步、控制频率和状态tokens
        # 最后horizon个token对应预测的动作序列
        x = x[:, -self.horizon:]  # (B, horizon, output_dim)
        
        return x