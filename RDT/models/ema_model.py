# Reference: DiffusionPolicy [https://github.com/real-stanford/diffusion_policy]

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    指数移动平均(Exponential Moving Average)模型权重管理类
    
    EMA是一种常用的模型权重平滑技术，通过维护模型参数的移动平均来提高模型的稳定性和泛化能力。
    在训练过程中，EMA模型的权重按照以下公式更新：
    ema_param = decay * ema_param + (1 - decay) * current_param
    
    其中decay是衰减因子，随着训练步数的增加而逐渐增大，最终趋于一个稳定值。
    """

    def __init__(self, model, update_after_step=0, inv_gamma=1.0, power=2/3, min_value=0.0, max_value=0.9999):
        """
        初始化EMA模型
        
        EMA预热机制说明（来自@crowsonkb的注释）：
        - 如果gamma=1且power=1，实现简单平均
        - gamma=1, power=2/3适合训练百万步以上的模型（在31.6K步达到0.999衰减因子，1M步达到0.9999）
        - gamma=1, power=3/4适合训练步数较少的模型（在10K步达到0.999衰减因子，215.4K步达到0.9999）
        
        Args:
            model: 需要进行EMA的原始模型
            update_after_step (int): 开始更新EMA的步数阈值，默认为0
            inv_gamma (float): EMA预热的倒数乘法因子，默认为1.0
            power (float): EMA预热的指数因子，默认为2/3
            min_value (float): EMA衰减率的最小值，默认为0.0
            max_value (float): EMA衰减率的最大值，默认为0.9999
        """
        
        # 创建EMA模型的副本，并设置为评估模式
        self.averaged_model = model
        self.averaged_model.eval()  # 设置为评估模式，禁用dropout等
        self.averaged_model.requires_grad_(False)  # 禁用梯度计算，节省内存

        # 保存EMA参数配置
        self.update_after_step = update_after_step  # 延迟更新的步数
        self.inv_gamma = inv_gamma  # 预热参数：倒数gamma
        self.power = power  # 预热参数：幂次
        self.min_value = min_value  # 衰减因子最小值
        self.max_value = max_value  # 衰减因子最大值

        # 初始化内部状态
        self.decay = 0.0  # 当前衰减因子
        self.optimization_step = 0  # 当前优化步数

    def get_decay(self, optimization_step):
        """
        计算指数移动平均的衰减因子
        
        衰减因子的计算公式：
        1. 首先计算有效步数：step = max(0, optimization_step - update_after_step - 1)
        2. 然后计算衰减值：value = 1 - (1 + step / inv_gamma)^(-power)
        3. 最后应用边界限制：max(min_value, min(value, max_value))
        
        这个公式实现了EMA的"预热"机制：
        - 在训练初期，衰减因子较小，新参数的权重更大
        - 随着训练进行，衰减因子逐渐增大，EMA变得更稳定
        
        Args:
            optimization_step (int): 当前优化步数
            
        Returns:
            float: 计算得到的衰减因子
        """
        # 计算有效步数（减去延迟步数）
        step = max(0, optimization_step - self.update_after_step - 1)
        
        # 使用预热公式计算衰减值
        # 公式：1 - (1 + step/inv_gamma)^(-power)
        value = 1 - (1 + step / self.inv_gamma)**(-self.power)

        # 如果还在延迟期内，返回0（不进行EMA更新）
        if step <= 0:
            return 0.0

        # 将衰减因子限制在[min_value, max_value]范围内
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()  # 禁用梯度计算，提高效率
    def step(self, new_model):
        """
        执行一步EMA更新
        
        这个方法遍历新模型和EMA模型的所有参数，根据不同的参数类型采用不同的更新策略：
        1. BatchNorm层：直接复制新参数（因为BN统计量需要保持当前状态）
        2. 不需要梯度的参数：直接复制新参数
        3. 需要梯度的参数：使用EMA公式更新
        
        Args:
            new_model: 当前训练的新模型，从中获取最新的参数值
        """
        # 计算当前步的衰减因子
        self.decay = self.get_decay(self.optimization_step)

        # 注释掉的代码是用于验证参数遍历一致性的调试代码
        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()  # 用于调试的数据指针集合
        
        # 同时遍历新模型和EMA模型的所有模块
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            # 只遍历每个模块的直接参数（不递归到子模块）
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                
                # 参数类型检查（字典类型参数不支持）
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')

                # 调试代码：收集参数的数据指针
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # 对于BatchNorm层，直接复制参数
                    # 原因：BN层的running_mean和running_var需要保持最新状态
                    # 而不是使用移动平均，否则会影响归一化效果
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                    
                elif not param.requires_grad:
                    # 对于不需要梯度的参数（如某些固定权重），直接复制
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                    
                else:
                    # 对于普通的可训练参数，使用EMA公式更新
                    # EMA公式：ema_param = decay * ema_param + (1 - decay) * new_param
                    
                    # 先将EMA参数乘以衰减因子
                    ema_param.mul_(self.decay)
                    # 再加上新参数乘以(1-衰减因子)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # 验证两种参数遍历方式的一致性（调试用）
        # assert old_all_dataptrs == all_dataptrs
        
        # 增加优化步数计数器
        self.optimization_step += 1