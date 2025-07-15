


          
用户想了解当执行 `bash eval.sh beat_block_hammer demo_clean RDT_demo_randomized 10000 0 0` 时，整个模型导入过程是如何实现的。我需要分析这个调用的参数传递和模型加载流程。
        
            
toolName: search_codebase
            
status: success
          
query: 分析 eval.sh 脚本中参数传递和 eval_policy.py 中模型导入的完整流程，特别是 get_model 函数如何根据参数加载模型
          
现在我需要查看具体的RDT模型导入实现，特别是RDT/deploy_policy.py中的get_model函数的完整实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/deploy_policy.py
          
现在我需要查看RDT模型类的具体实现，了解模型是如何被初始化和加载的。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/model/__init__.py
          
让我查看RDT目录下的文件结构，找到模型相关的文件。
        
            
toolName: view_folder
            
status: success
          
/Users/mac/Downloads/RoboTwin/policy/RDT
          
现在我需要查看model.py文件，这应该包含RDT模型类的定义。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/model.py
          
继续查看RDT模型的get_action方法和make_policy方法的实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/model.py
          
现在我需要查看create_model函数的实现，了解RDT模型是如何被创建的。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/scripts/agilex_model.py
          
继续查看agilex_model.py文件的剩余部分，特别是step方法的实现。
        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/mac/Downloads/RoboTwin/policy/RDT/scripts/agilex_model.py
          
## RDT模型导入过程完整分析

当执行 `bash eval.sh beat_block_hammer demo_clean RDT_demo_randomized 10000 0 0` 时，整个模型导入过程如下：

### 1. 参数传递流程

**eval.sh脚本参数解析：**
- `task_name=beat_block_hammer` (任务名称)
- `task_config=demo_clean` (任务配置)
- `model_name=RDT_demo_randomized` (模型名称)
- `checkpoint_id=10000` (检查点ID)
- `seed=0` (随机种子)
- `gpu_id=0` (GPU设备ID)

**调用eval_policy.py：**
```bash
python script/eval_policy.py --config policy/RDT/deploy_policy.yml \
    --overrides \
    --task_name beat_block_hammer \
    --task_config demo_clean \
    --ckpt_setting RDT_demo_randomized \
    --seed 0 \
    --checkpoint_id 10000 \
    --policy_name RDT
```

### 2. 模型导入的关键位置

**第一阶段：动态函数创建（eval_policy.py:191行）**
```python
get_model = eval_function_decorator(policy_conda_env, policy_name, "get_model")
```
- 通过 `eval_function_decorator` 创建模型获取函数
- 支持本地导入或conda环境远程执行

**第二阶段：模型实例化（eval_policy.py:299行）**
```python
model = get_model(usr_args)
```
- 调用动态创建的函数，传入用户参数
- 实际执行 `policy/RDT/deploy_policy.py` 中的 `get_model` 函数

### 3. RDT模型具体加载过程

**deploy_policy.py中的get_model函数：**
```python
def get_model(usr_args):
    model_name = usr_args["ckpt_setting"]  # "RDT_demo_randomized"
    checkpoint_id = usr_args["checkpoint_id"]  # "10000"
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"], 
        usr_args["rdt_step"]
    )
    
    # 创建RDT实例
    rdt = RDT(
        os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt"
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    )
    return rdt
```

### 4. RDT类初始化过程

**模型文件路径构建：**
```
checkpoints/RDT_demo_randomized/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt
```

**RDT.__init__方法执行：**
1. **配置设置**：设置episode长度、状态维度、chunk大小、相机名称等
2. **策略模型加载**：调用 `make_policy()` 方法
3. **语言编码器初始化**：调用 `set_language_embed()` 加载T5文本编码器
4. **观察窗口初始化**：设置为None，后续动态创建

### 5. 核心模型组件加载

**make_policy方法：**
```python
def make_policy(self, args):
    # 加载配置文件
    with open(args["config_path"], "r") as fp:
        config_base_yaml = yaml.safe_load(fp)
    
    # 设置视觉编码器路径
    pretrained_vision_encoder_name_or_path = os.path.join(
        self.global_path, "weights/RDT/siglip-so400m-patch14-384"
    )
    
    # 创建RoboticDiffusionTransformerModel
    model = create_model(
        args=args["config"],
        dtype=torch.bfloat16,
        pretrained=args["pretrained_model_name_or_path"],
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args["ctrl_freq"],
    )
    return model
```

**RoboticDiffusionTransformerModel初始化：**
1. **视觉编码器**：加载SiglipVisionTower
2. **策略网络**：创建RDTRunner实例
3. **权重加载**：从checkpoint文件加载预训练权重
4. **设备配置**：将模型移动到指定GPU并设置数据类型

### 6. 权重加载机制

**load_pretrained_weights方法：**
```python
def load_pretrained_weights(self, pretrained=None):
    if pretrained is None:
        return
    print(f"Loading weights from {pretrained}")
    filename = os.path.basename(pretrained)
    if filename.endswith(".pt"):
        checkpoint = torch.load(pretrained)
        self.policy.load_state_dict(checkpoint["module"])
    elif filename.endswith(".safetensors"):
        from safetensors.torch import load_model
        load_model(self.policy, pretrained)
```

### 7. 模型推理准备

**模型设置为评估模式：**
- 策略网络：`self.policy.eval()`
- 视觉编码器：`self.vision_model.eval()`
- 数据类型转换：转换为bfloat16精度
- 设备迁移：移动到指定GPU

### 总结

整个模型导入过程是一个层次化的加载机制：
1. **参数传递**：从shell脚本到Python脚本
2. **动态导入**：通过装饰器模式支持不同环境
3. **模型构建**：分层创建RDT、视觉编码器、文本编码器
4. **权重加载**：从checkpoint文件恢复预训练权重
5. **推理准备**：设置评估模式和设备配置

这种设计使得模型加载过程既灵活又模块化，支持不同的部署环境和配置需求。
        