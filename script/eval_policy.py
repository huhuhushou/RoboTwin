# 机器人策略评估系统
# 该脚本用于评估机器人在不同任务环境中的策略表现
import sys
import os
import subprocess

# 添加必要的路径到Python搜索路径中
sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

# 获取当前文件的绝对路径和父目录
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    """
    任务类装饰器函数
    
    参数:
        task_name (str): 任务名称
    
    返回:
        env_instance: 任务环境实例
    
    功能:
        - 动态导入指定的任务模块
        - 创建并返回任务环境实例
        - 如果任务不存在则抛出异常
    """
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    """
    策略评估函数装饰器
    
    参数:
        policy_name (str): 策略模块名称
        model_name (str): 模型函数名称
        conda_env (str, optional): Conda环境名称，用于外部环境执行
    
    返回:
        function: 策略评估函数
    
    功能:
        - 如果没有指定conda环境，直接导入并返回函数
        - 如果指定了conda环境，创建外部执行函数，通过子进程在指定环境中运行
    """
    if conda_env is None:
        # 直接在当前环境中导入和执行
        try:
            policy_model = importlib.import_module(policy_name)
            return getattr(policy_model, model_name)
        except ImportError as e:
            raise e
    else:
        # 在外部conda环境中执行
        def external_eval(*args, **kwargs):
            import pickle
            import tempfile
            import os

            # 创建临时目录进行数据传递
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.pkl")
                output_path = os.path.join(tmpdir, "output.pkl")

                # 将输入参数序列化保存
                with open(input_path, "wb") as f:
                    pickle.dump((policy_name, model_name, args, kwargs), f)

                # 创建shell脚本在指定conda环境中执行
                script = f"""
source ~/.bashrc
conda activate {conda_env}
python run_remote_model.py "{input_path}" "{output_path}"
"""

                # 执行外部脚本
                subprocess.run(script, shell=True, check=True, executable="/bin/bash")

                # 读取执行结果
                with open(output_path, "rb") as f:
                    result = pickle.load(f)
                return result

        return external_eval


def get_camera_config(camera_type):
    """
    获取相机配置参数
    
    参数:
        camera_type (str): 相机类型
    
    返回:
        dict: 相机配置字典
    
    功能:
        - 从配置文件中读取指定类型相机的配置参数
        - 验证配置文件存在性和相机类型有效性
    """
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    """
    获取机器人实体配置
    
    参数:
        robot_file (str): 机器人配置文件路径
    
    返回:
        dict: 机器人实体配置字典
    
    功能:
        - 从指定路径读取机器人的配置信息
        - 包含关节名称、控制参数等
    """
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    """
    主函数 - 评估系统的入口点
    
    参数:
        usr_args (dict): 用户配置参数
    
    功能:
        - 解析和设置评估参数
        - 配置机器人实体和相机
        - 创建保存目录
        - 执行策略评估
        - 保存评估结果
    """
    # 获取当前时间戳用于结果保存
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 从用户参数中提取关键配置
    task_name = usr_args["task_name"]           # 任务名称
    task_config = usr_args["task_config"]       # 任务配置文件名
    ckpt_setting = usr_args["ckpt_setting"]     # 检查点设置
    policy_name = usr_args["policy_name"]       # 策略名称
    instruction_type = usr_args["instruction_type"]  # 指令类型
    
    # 初始化保存路径变量
    save_dir = None
    video_save_dir = None
    video_size = None

    # 获取策略conda环境配置（如果有）
    policy_conda_env = usr_args.get("policy_conda_env", None)

    # 创建模型获取函数
    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    # 读取任务配置文件
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 更新配置参数
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    # 获取机器人实体配置
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        """获取机器人实体文件路径"""
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    # 读取相机配置
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 设置头部相机配置
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # 配置双臂机器人设置
    if len(embodiment_type) == 1:
        # 单一实体类型，用于双臂
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        # 不同实体类型，分别配置左右臂
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]  # 实体间距离
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    # 获取左右臂的具体配置
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # 生成实体名称字符串
    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # 创建结果保存目录
    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 如果启用视频记录，设置视频保存路径
    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # 输出配置信息到控制台
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    # 创建任务环境实例
    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    
    # 设置左右臂关节维度
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # 获取随机种子
    seed = usr_args["seed"]

    # 计算起始种子和初始化评估参数
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100  # 测试轮数
    topk = 1

    # 获取模型实例
    model = get_model(usr_args)
    
    # 执行策略评估
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   policy_conda_env=policy_conda_env)
    suc_nums.append(suc_num)

    # 计算top-k成功率
    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    # 保存评估结果到文件
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None):
    """
    策略评估核心函数
    
    参数:
        task_name (str): 任务名称
        TASK_ENV: 任务环境实例
        args (dict): 配置参数
        model: 策略模型
        st_seed (int): 起始随机种子
        test_num (int): 测试次数
        video_size (str): 视频尺寸
        instruction_type (str): 指令类型
        policy_conda_env (str): 策略conda环境
    
    返回:
        tuple: (最终种子, 成功次数)
    
    功能:
        - 循环执行指定次数的策略评估
        - 记录成功率和失败情况
        - 生成视频记录（如果启用）
        - 处理各种异常情况
    """
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # 初始化评估状态
    expert_check = True  # 专家检查标志
    TASK_ENV.suc = 0     # 成功次数
    TASK_ENV.test_num = 0 # 测试次数

    now_id = 0           # 当前episode ID
    succ_seed = 0        # 成功的种子数
    suc_test_seed_list = [] # 成功测试的种子列表

    # 获取策略相关的函数
    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)
    reset_func = eval_function_decorator(policy_name, "reset_model", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]  # 缓存清理频率

    args["eval_mode"] = True

    # 主评估循环
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0  # 临时关闭渲染以加速专家检查

        # 专家检查阶段 - 验证任务是否可以成功完成
        if expert_check:
            try:
                # 设置演示环境
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()  # 执行一次演示
                TASK_ENV.close_env()
            except UnStableError as e:
                # 处理不稳定错误
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # 处理其他异常
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        # 检查专家是否成功完成任务
        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # 重新设置环境进行策略评估
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        
        # 生成episode描述和指令
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # 设置语言指令

        # 设置视频录制（如果启用）
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",                    # 覆盖输出文件
                    "-loglevel", "error",    # 只显示错误信息
                    "-f", "rawvideo",        # 输入格式为原始视频
                    "-pixel_format", "rgb24", # 像素格式
                    "-video_size", video_size, # 视频尺寸
                    "-framerate", "10",      # 帧率
                    "-i", "-",               # 从stdin读取
                    "-pix_fmt", "yuv420p",   # 输出像素格式
                    "-vcodec", "libx264",    # 视频编码器
                    "-crf", "23",            # 质量参数
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        # 执行策略评估
        succ = False
        reset_func(model)  # 重置模型状态
        
        # 主要的action执行循环
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()  # 获取观察
            eval_func(TASK_ENV, model, observation)  # 执行策略
            if TASK_ENV.eval_success:
                succ = True
                break
        
        # 结束视频录制
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # 记录结果
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")  # 绿色成功信息
        else:
            print("\033[91mFail!\033[0m")     # 红色失败信息

        now_id += 1
        
        # 清理环境（定期清理缓存）
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        # 打印当前进度信息
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    """
    解析命令行参数和配置文件
    
    返回:
        dict: 合并后的配置字典
    
    功能:
        - 解析命令行参数，包括配置文件路径和覆盖参数
        - 读取YAML配置文件
        - 应用命令行覆盖参数
        - 返回最终的配置字典
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)  # 配置文件路径
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)  # 覆盖参数
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 解析覆盖参数
    def parse_override_pairs(pairs):
        """
        解析覆盖参数对
        格式: --key1 value1 --key2 value2
        """
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")  # 移除--前缀
            value = pairs[i + 1]
            try:
                value = eval(value)  # 尝试评估为Python对象
            except:
                pass  # 如果失败则保持字符串
            override_dict[key] = value
        return override_dict

    # 应用覆盖参数
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    """
    主程序入口
    
    功能:
        - 导入测试渲染模块并执行测试
        - 解析配置参数
        - 启动主评估流程
    """
    from test_render import Sapien_TEST
    Sapien_TEST()  # 执行Sapien渲染测试

    usr_args = parse_args_and_config()  # 解析配置

    main(usr_args)  # 启动主程序