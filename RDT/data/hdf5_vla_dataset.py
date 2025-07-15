import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, model_config_path) -> None:
        # 从模型配置中读取HDF5数据集目录路径
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        HDF5_DIR = model_config["data_path"]  # 指向第一个代码的输出目录
        self.DATASET_NAME = "agilex"          # 数据集名称标识
        
        # 递归搜索所有HDF5文件（每个文件对应一个episode）
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)  # 收集所有episode文件路径
        
        # 加载训练配置参数
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]      # 动作序列长度
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]  # 图像历史帧数
        self.STATE_DIM = config["common"]["state_dim"]               # 统一状态空间维度
        
        # 计算每个episode的采样权重（基于episode长度）
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res["state"].shape[0] if valid else 0
            episode_lens.append(_len)
        # 长episode有更高的采样概率
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = (self.parse_hdf5_file(file_path)
                             if not state_only else self.parse_hdf5_file_state_only(file_path))
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, "r") as f:
            # 读取第一个代码生成的基础数据
            qpos = f["observations"]["qpos"][:]           # 状态序列 (T-1, state_dim)
            left_arm_dim = f["observations"]["left_arm_dim"][:]   # 左臂维度信息
            right_arm_dim = f["observations"]["right_arm_dim"][:] # 右臂维度信息
            num_steps = qpos.shape[0]                     # 总时间步数
            
            # 跳过前几个静止步骤（可选优化）
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])        # 计算与初始状态的差异
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]  # 找到开始运动的时刻
            first_idx = indices[0] if len(indices) > 0 else 0
            
            # 随机采样一个时间步作为当前状态
            step_id = np.random.randint(first_idx - 1, num_steps)
            
            # 加载语言指令（使用预计算的嵌入）
            dir_path = os.path.dirname(file_path)
            instructions_path = os.path.join(dir_path, "instructions")
            instructions_names = []
            for filename in os.listdir(instructions_path):
                if filename.endswith(".pt"):              # 查找.pt格式的预计算嵌入
                    instructions_names.append(os.path.join(instructions_path, filename))
            instruction = np.random.choice(instructions_names)  # 随机选择一个指令
            
            # 组装元数据
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
            }
            
            # 数据标准化（这里是占位符，实际可能需要具体的标准化策略）
            qpos = qpos / np.array([[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])
            target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / np.array(
                [[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])
            
            # 解析状态和动作
            state = qpos[step_id:step_id + 1]            # 当前状态 (1, state_dim)
            state_std = np.std(qpos, axis=0)             # 状态标准差
            state_mean = np.mean(qpos, axis=0)           # 状态均值  
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))  # 状态范数
            actions = target_qpos                        # 动作序列
            
            # 动作序列填充（如果不足CHUNK_SIZE）
            if actions.shape[0] < self.CHUNK_SIZE:
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                ], axis=0)
            
            # 将机器人特定的状态映射到统一状态空间
            def fill_in_state(values):
                # 根据机器人配置确定状态向量中各关节的位置
                UNI_STATE_INDICES = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] 
                    for i in range(left_arm_dim[0])] +                    # 左臂关节
                    [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +         # 左夹爪
                    [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] 
                    for i in range(right_arm_dim[0])] +                   # 右臂关节
                    [STATE_VEC_IDX_MAPPING["right_gripper_open"]]          # 右夹爪
                )
                # 创建统一维度的零向量，然后填入实际值
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            # 应用统一状态空间映射
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))  # 有效性指示器
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_state(actions)
            
            # 解析图像数据
            def parse_img(key):
                """从JPEG编码数据中解析图像历史"""
                imgs = []
                # 获取历史帧：从当前时刻向前IMG_HISORY_SIZE帧
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    img_bits = f["observations"]["images"][key][i]  # 获取JPEG编码数据
                    img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                    imgs.append(img)
                imgs = np.stack(imgs)
                
                # 如果历史帧不足，用第一帧填充
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs,
                    ], axis=0)
                return imgs
            
            # 处理三个摄像头的图像（对应第一个代码的三个摄像头）
            cam_high = parse_img("cam_high")              # 头部摄像头
            cam_left_wrist = parse_img("cam_left_wrist")  # 左手腕摄像头
            cam_right_wrist = parse_img("cam_right_wrist") # 右手腕摄像头
            
            # 计算图像有效性掩码
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # 返回完整的训练样本
            return True, {
                "meta": meta,
                "state": state,                    # 当前状态
                "state_std": state_std,            # 统计信息
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,                # 动作序列
                "state_indicator": state_indicator, # 状态有效性
                "cam_high": cam_high,              # 图像数据
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            left_arm_dim = f["observations"]["left_arm_dim"][:]
            right_arm_dim = f["observations"]["right_arm_dim"][:]

            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            # return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # Rescale gripper to [0, 1]
            qpos = qpos / np.array([[1 for i in range(left_arm_dim[0] + right_arm_dim[0] + 2)]])
            target_qpos = f["action"][:] / np.array([[1 for i in range(left_arm_dim[0] + right_arm_dim[0] + 2)]])

            # Parse the state and action
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                     for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                    [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                     for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            action = fill_in_state(action)

            # Return the resulting sample            return True, {"state": state, "action": action}



if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
