import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt


def get_clean_item(chunk_dir):
    """
    获取chunk中干净(未被使用)的数据项索引
    
    Args:
        chunk_dir (str): chunk目录路径
        
    Returns:
        list: 干净数据项的索引列表
    """
    dirty_bit = read_dirty_bit(chunk_dir)
    # 返回dirty_bit为0的位置索引(即干净的数据项)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """
    保存dirty bit到chunk目录
    dirty bit用于标记哪些数据项已被使用，避免重复训练
    
    Args:
        chunk_dir (str): chunk目录路径
        dirty_bit (np.ndarray): dirty bit数组
    """
    time_stmp = time.time()
    # 重试机制，最多尝试10秒
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()  # 获取写锁，确保线程安全
            with open(file_path, "wb") as file:
                file.write(dirty_bit.tobytes())  # 写入二进制数据
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue  # 出错时继续重试
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """
    从chunk目录读取dirty bit
    
    Args:
        chunk_dir (str): chunk目录路径
        
    Returns:
        np.ndarray: dirty bit数组
    """
    time_stmp = time.time()
    # 重试机制，最多尝试10秒
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()  # 获取读锁
            with open(file_path, "rb") as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


class VLAConsumerDataset(Dataset):
    """
    视觉-语言-动作(Vision-Language-Action)数据集，用于监督学习训练
    该数据集从缓冲区目录加载数据，支持多种配置和数据增强
    """

    def __init__(
        self,
        model_config_path,        # 模型配置文件路径
        config,                   # 数据集配置
        tokenizer,                # 文本tokenizer
        image_processor,          # 图像处理器
        num_cameras,              # 摄像头数量
        img_history_size,         # 图像历史长度
        image_size=None,          # 图像尺寸
        auto_adjust_image_brightness=False,  # 是否自动调整图像亮度
        image_aug=False,          # 是否进行图像增强
        dataset_type="pretrain",  # 数据集类型(pretrain/finetune)
        cond_mask_prob=0.1,      # 条件掩码概率
        cam_ext_mask_prob=-1.0,  # 外部摄像头掩码概率
        state_noise_snr=None,    # 状态噪声信噪比
        use_hdf5=False,          # 是否使用HDF5格式
        use_precomp_lang_embed=False,  # 是否使用预计算的语言嵌入
    ):
        super(VLAConsumerDataset, self).__init__()

        # 加载每个数据集的控制频率配置
        with open("configs/dataset_control_freq.json", "r") as fp:
            self.control_freq = json.load(fp)
            
        # 根据数据集类型加载数据集名称列表
        dataset_names_cfg = ("configs/pretrain_datasets.json"
                             if dataset_type == "pretrain" else "configs/finetune_datasets.json")
        with open(dataset_names_cfg, "r") as file:
            DATASET_NAMES = json.load(file)
            
        # 创建数据集名称和ID之间的映射关系
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        # 保存配置参数
        self.image_processor = image_processor
        self.model_config_path = model_config_path
        self.buffer_dir = config["buf_path"]           # 缓冲区路径
        self.num_chunks = config["buf_num_chunks"]     # chunk数量
        self.chunk_size = config["buf_chunk_size"]     # 每个chunk的大小
        self.tokenizer_max_length = config["tokenizer_max_length"]  # tokenizer最大长度
        self.image_aspect_ratio = config["image_aspect_ratio"]      # 图像宽高比处理方式
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob           # 条件掩码概率，用于数据增强
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        
        # 如果使用HDF5格式，初始化HDF5数据集
        self.hdf5_dataset = None
        if use_hdf5:
            self.hdf5_dataset = HDF5VLADataset(self.model_config_path)
            
        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            # 加载空的语言嵌入，用于掩码时使用
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")

        # 加载数据集统计信息
        with open("configs/dataset_stat.json", "r") as f:
            dataset_stat = json.load(f)
        self.dataset_stat = dataset_stat

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug

        # 缓存最后加载的内容，用于错误恢复
        self.last_content = None
        self.last_meta = None

    def get_dataset_name2id(self):
        """获取数据集名称到ID的映射"""
        return self.dataset_name2id

    def get_dataset_id2name(self):
        """获取数据集ID到名称的映射"""
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        """将迭代器按两两配对的方式分组"""
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def _load_data_from_chunk(chunk_dir, chunk_item_idx):
        """
        从指定chunk中加载数据
        
        Args:
            chunk_dir (str): chunk目录路径
            chunk_item_idx (int): chunk内的数据项索引
            
        Returns:
            tuple: (json内容, 元数据)
        """
        time_stmp = time.time()
        # 重试机制
        while time.time() - time_stmp < 10.0:
            try:
                locks = []
                
                # 加载JSON内容
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "r") as file:
                    json_content = json.load(file)
                lock.release_lock()
                
                # 加载数据文件(npz格式)
                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "rb") as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()
                
                return json_content, meta
            except KeyboardInterrupt:
                for lock in locks:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                for lock in locks:
                    lock.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def __len__(self) -> int:
        """返回数据集长度"""
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size

    def _safe_load(self, index):
        """
        安全地加载数据，包含错误处理和重试机制
        
        Args:
            index (int): 数据索引
            
        Returns:
            tuple: 加载的数据
        """
        read_chunk_item_indices = []
        # 从随机chunk开始搜索
        read_chunk_idx = index // self.chunk_size
        
        # 找到有干净数据项的chunk
        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
            except BaseException as e:
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            # 循环到下一个chunk
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

        # 选择一个随机的干净数据项
        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]

        # 修改dirty bit，标记该数据项已被使用
        try:
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1  # 标记为已使用
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            print("Error catched when modifying the dirty bit:", e)
            traceback.print_exc()

        # 加载样本数据
        try:
            content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            # 如果加载失败，返回最后成功加载的数据，提高鲁棒性
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            dict: 包含所有必要信息的数据字典
        """
        # 为了鲁棒性，持续尝试直到成功加载数据
        while True:
            data_dict = None
            try:
                # 根据配置选择数据源
                if self.use_hdf5:
                    # 从HDF5数据集加载
                    res = self.hdf5_dataset.get_item()
                    content = res["meta"]
                    states = res["state"]
                    actions = res["actions"]
                    state_elem_mask = res["state_indicator"]
                    image_metas = [
                        res["cam_high"],
                        res["cam_high_mask"],
                        res["cam_right_wrist"],
                        res["cam_right_wrist_mask"],
                        res["cam_left_wrist"],
                        res["cam_left_wrist_mask"],
                    ]
                    state_std = res["state_std"]
                    state_mean = res["state_mean"]
                    state_norm = res["state_norm"]
                else:
                    # 从缓冲区加载
                    (
                        content,
                        _,
                        states,
                        _,
                        actions,
                        _,
                        state_elem_mask,
                        *image_metas,
                        state_std,
                        state_mean,
                        state_norm,
                    ) = self._safe_load(index)

                # 构建数据字典
                data_dict = {}
                data_dict["dataset_name"] = content["dataset_name"]
                data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
                
                # 控制频率，可能被随机掩码
                data_dict["ctrl_freq"] = (self.control_freq[data_dict["dataset_name"]]
                                          if random.random() > self.cond_mask_prob else 0)

                # 可选地向状态添加噪声
                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0,
                        state_std / np.sqrt(10**(self.state_noise_snr / 10)),
                        states.shape,
                    )
                    
                # 获取数据集的平均状态，用于掩码
                ds_state_mean = np.array(self.dataset_stat[data_dict["dataset_name"]]["state_mean"])
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                
                # 随机用平均状态掩码状态数据
                data_dict["states"] = (states if random.random() > self.cond_mask_prob else ds_state_mean)
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = (state_elem_mask if random.random() > self.cond_mask_prob else
                                                np.zeros_like(state_elem_mask))

                # 该步骤所属episode的统计信息
                data_dict["state_norm"] = state_norm

                # 处理图像数据
                # 创建背景图像，用于替换无效图像和随机掩码
                background_color = np.array(
                    [int(x * 255) for x in self.image_processor.image_mean],
                    dtype=np.uint8,
                ).reshape(1, 1, 3)
                background_image = (np.ones(
                    (
                        self.image_processor.size["height"],
                        self.image_processor.size["width"],
                        3,
                    ),
                    dtype=np.uint8,
                ) * background_color)

                # 将图像元数据按(图像, 掩码)对分组
                image_metas = list(self.pairwise(image_metas))
                
                # 设置每个摄像头的掩码概率
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob  # 外部摄像头特殊处理

                # 重新排列图像：历史帧 × 摄像头
                rearranged_images = []
                for i in range(self.img_history_size):      # 遍历历史帧
                    for j in range(self.num_cameras):       # 遍历摄像头
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        
                        # 检查图像是否有效且不被掩码
                        if (valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j])):
                            rearranged_images.append((image, True))
                        else:
                            # 使用背景图像替换
                            rearranged_images.append((background_image.copy(), False))

                # 预处理图像
                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    
                    # 调整图像大小
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)

                    # 自动调整亮度
                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:  # 如果图像太暗
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    # 图像增强(50%概率应用)
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice(["corrput_only", "color_only", "both"])
                        
                        # 颜色增强
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5,
                                                           hue=0.03)(image)
                        # 图像损坏增强
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                    # 处理图像宽高比
                    if self.image_aspect_ratio == "pad":
                        def expand2square(pil_img, background_color):
                            """将图像扩展为正方形，用背景色填充"""
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    
                    # 使用图像处理器进行最终预处理
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                    preprocessed_images.append(image)
                    
                data_dict["images"] = preprocessed_images

                # 处理语言指令
                if self.use_precomp_lang_embed:
                    # 使用预计算的语言嵌入
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = (torch.load(content["instruction"])
                                               if random.random() > self.cond_mask_prob else self.empty_lang_embed)
                else:
                    # 使用tokenizer处理指令
                    instruction = (content["instruction"] if random.random() > self.cond_mask_prob else "")
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]

                    # 检查指令长度是否超过限制
                    assert (
                        len(data_dict["input_ids"]) <= self.tokenizer_max_length
                    ), f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."

                # 将numpy数组转换为torch张量
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                # 最终检查：确保没有numpy数组残留
                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

                return data_dict
                
            except BaseException as e:
                # 错误处理：打印错误信息并重试
                if data_dict is not None:
                    print(
                        f"Error catched when processing sample from {data_dict.get('dataset_name')}:",
                        e,
                    )
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # 尝试下一个索引
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDataset(object):
    """
    VLA数据集的数据整理器，用于将多个样本整理成批次
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个实例整理成一个批次
        
        Args:
            instances (Sequence[Dict]): 样本实例列表
            
        Returns:
            Dict[str, torch.Tensor]: 批次数据字典
        """
        # 初始化批次字典
        batch = {
            "states": [],              # 状态列表
            "actions": [],             # 动作列表  
            "state_elem_mask": [],     # 状态元素掩码列表
            "state_norm": [],          # 状态标准化信息列表
            "images": [],              # 图像列表
            "data_indices": [],        # 数据集索引列表
            "ctrl_freqs": [],          # 控制频率列表
        }
        input_ids = []               # 输入ID列表(tokenizer输出)
        lang_embeds = []             # 语言嵌入列表
        lang_embed_lens = []         # 语言嵌入长度列表

        # 处理每个实例
        for instance in instances:
            # 确保关键数据转换为tensor格式
            keys_to_check = [
                "states",
                "actions", 
                "state_elem_mask",
                "state_norm",
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # 处理语言数据(两种方式：token IDs 或 预计算嵌入)
            if "input_ids" in instance:
                input_ids.append(instance["input_ids"])
            else:
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])

            # 处理图像、数据索引和控制频率
            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["data_indices"].append(instance["data_idx"])
            batch["ctrl_freqs"].append(instance["ctrl_freq"])

        # 将列表堆叠成张量
        keys_to_stack = ["states", "actions", "state_elem_mask", "state_norm", "images"]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

        # 处理语言数据的填充和注意力掩码
        if len(input_ids) > 0:
            # 使用tokenizer的填充token进行填充
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                        batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
            batch["input_ids"] = input_ids
            # 创建注意力掩码(非填充位置为True)
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            # 处理预计算的语言嵌入
            lang_embeds = torch.nn.utils.rnn.pad_sequence(lang_embeds, batch_first=True, padding_value=0)
            input_lang_attn_mask = torch.zeros(lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            # 根据原始长度设置注意力掩码
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask

        return batch