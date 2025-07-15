import sys

# 将当前目录添加到Python路径中，以便导入自定义模块
sys.path.append("./")

import os
import h5py           # 用于处理HDF5格式文件
import numpy as np
import pickle
import cv2            # OpenCV图像处理库
import argparse       # 命令行参数解析
import yaml           # YAML配置文件解析
from scripts.encode_lang_batch_once import encode_lang  # 导入语言编码函数


def load_hdf5(dataset_path):
    """
    从HDF5文件中加载机器人数据
    
    Args:
        dataset_path (str): HDF5数据集文件路径
        
    Returns:
        tuple: 包含左右手臂和夹爪数据，以及图像字典
            - left_gripper: 左手夹爪数据
            - left_arm: 左手臂关节数据  
            - right_gripper: 右手夹爪数据
            - right_arm: 右手臂关节数据
            - image_dict: 包含各个摄像头图像数据的字典
    """
    # 检查数据集文件是否存在
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    # 打开HDF5文件并读取数据
    with h5py.File(dataset_path, "r") as root:
        # 读取左手夹爪和手臂的关节动作数据
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        # 读取右手夹爪和手臂的关节动作数据
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        
        # 读取所有摄像头的图像数据
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            # 获取每个摄像头的RGB图像数据
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    """
    对图像列表进行JPEG编码并填充到相同长度
    
    Args:
        imgs (list): 图像数组列表
        
    Returns:
        tuple: 
            - encode_data: 编码后的JPEG数据列表
            - max_len: 最大编码长度
    """
    encode_data = []     # 存储编码后的数据
    padded_data = []     # 存储填充后的数据
    max_len = 0          # 记录最大编码长度
    
    # 第一步：将所有图像编码为JPEG格式
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])  # 将图像编码为JPEG
        jpeg_data = encoded_image.tobytes()                     # 转换为字节数据
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))                  # 更新最大长度
    
    # 第二步：将所有编码数据填充到相同长度（使用零字节填充）
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    
    return encode_data, max_len


def get_task_config(task_name):
    """
    从YAML文件中加载任务配置
    
    Args:
        task_name (str): 任务名称
        
    Returns:
        dict: 任务配置字典
    """
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    """
    主要的数据转换函数：将原始数据转换为训练格式
    
    Args:
        path (str): 原始数据路径
        episode_num (int): 要处理的episodes数量
        save_path (str): 保存处理后数据的路径
        
    Returns:
        int: 处理成功的episodes数量
    """
    begin = 0  # 计数器
    floders = os.listdir(path)  # 获取目录下所有文件
    
    # 检查是否有足够的数据
    assert episode_num <= len(floders), "data num not enough"

    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 处理每个episode
    for i in range(episode_num):
        print(f"Processing episode {i}...")
        
        # 加载当前episode的数据
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (
            load_hdf5(os.path.join(path, f"episode{i}.hdf5")))
        
        # 初始化存储列表
        qpos = []              # 关节位置（状态）
        actions = []           # 动作序列
        cam_high = []          # 头部摄像头图像
        cam_right_wrist = []   # 右手腕摄像头图像
        cam_left_wrist = []    # 左手腕摄像头图像
        left_arm_dim = []      # 左手臂维度
        right_arm_dim = []     # 右手臂维度

        last_state = None
        
        # 处理时间序列中的每一帧
        for j in range(0, left_gripper_all.shape[0]):
            # 获取当前帧的关节数据
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            # 将所有关节数据concatenate成一个状态向量
            # 顺序：左手臂 + 左夹爪 + 右手臂 + 右夹爪
            '''
            state = [left_arm[0], left_arm[1], ..., left_arm[n-1],    # 左手臂关节角度
                    left_gripper,                                      # 左夹爪开合度
                    right_arm[0], right_arm[1], ..., right_arm[m-1],  # 右手臂关节角度  
                    right_gripper]                                     # 右夹爪开合度
            '''
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
            state = state.astype(np.float32)


            '''
            qpos[i] 是时刻 i 的机器人状态观察
            action[i] 是在状态 qpos[i] 下要执行的动作（即下一时刻的目标状态）
            images[i] 是时刻 i 对应的三个摄像头图像

            state_dim = left_arm_joints + 1 + right_arm_joints + 1
                      = left_arm_dim + left_gripper + right_arm_dim + right_gripper

            '''
            # 除了最后一帧，都作为观察状态（qpos）
            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                # 处理头部摄像头图像
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))  # 调整图像大小
                cam_high.append(camera_high_resized)

                # 处理右手腕摄像头图像
                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                # 处理左手腕摄像头图像
                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            # 除了第一帧，都作为动作（action）
            if j != 0:
                action = state  # 当前状态作为要执行的动作
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])   # 记录左手臂维度
                right_arm_dim.append(right_arm.shape[0]) # 记录右手臂维度

        # 为当前episode创建保存目录
        if not os.path.exists(os.path.join(save_path, f"episode_{i}")):
            os.makedirs(os.path.join(save_path, f"episode_{i}"))
        
        # 定义HDF5保存路径
        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        # 将处理后的数据保存为HDF5格式
        with h5py.File(hdf5path, "w") as f:
            # 保存动作序列
            f.create_dataset("action", data=np.array(actions))
            
            # 创建observations组
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))                    # 关节位置
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))    # 左手臂维度
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))  # 右手臂维度
            
            # 创建images组并保存编码后的图像
            image = obs.create_group("images")
            
            # 对三个摄像头的图像进行编码
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            
            # 保存编码后的图像数据（使用字符串类型存储二进制数据）
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"Episode {i} processed successfully!")

    return begin


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process robot training episodes.")
    parser.add_argument("task_name", type=str, help="Name of the task")
    parser.add_argument("task_config", type=str, help="Task configuration name") 
    parser.add_argument("expert_data_num", type=int, help="Number of expert episodes to process")
    args = parser.parse_args()

    # 获取命令行参数
    task_name = args.task_name
    task_config = args.task_config  
    expert_data_num = args.expert_data_num

    # 构建数据加载路径
    load_dir = os.path.join("../../data", str(task_name), str(task_config), "data")

    print(f"Reading data from path: {load_dir}")
    
    # 执行数据转换
    begin = data_transform(
        load_dir,
        expert_data_num,
        f"./processed_data/{task_name}-{task_config}-{expert_data_num}",
    )
    
    # 处理语言指令数据
    tokenizer, text_encoder = None, None
    for idx in range(expert_data_num):
        print(f"Processing Language instruction for episode {idx}", end="\r")
        
        # 构建指令文件路径
        data_file_path = (f"../../data/{task_name}/{task_config}/instructions/episode{idx}.json")
        # 构建目标保存路径
        target_dir = (f"processed_data/{task_name}-{task_config}-{expert_data_num}/episode_{idx}")
        
        # 编码语言指令并保存
        tokenizer, text_encoder = encode_lang(
            DATA_FILE_PATH=data_file_path,    # 指令数据文件路径
            TARGET_DIR=target_dir,            # 目标保存目录
            GPU=0,                            # 使用的GPU编号
            desc_type="seen",                 # 描述类型
            tokenizer=tokenizer,              # 复用tokenizer以提高效率
            text_encoder=text_encoder,        # 复用text_encoder以提高效率
        )
    
    print(f"\nData processing completed! Processed {begin} episodes successfully.")