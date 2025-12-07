# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2024/6/26 上午10:25
@Author  : H-Tenets
@File    : SetRunPath.py
@Software: PyCharm 
@Git-v   : 
"""
import os
import platform
import sys

import toml

TrainConfig = toml.load("TrainConfig.toml")

def setRunPath():
    """
    Dynamically set the runtime path according to the operating system。
    """
    if platform.system() == 'Windows':
        # local path
        code_path = TrainConfig['path']['local']['code_root']
        root = TrainConfig['path']['local']['code_root']
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    elif platform.system() == 'Linux':
        # remote path
        code_path = TrainConfig['path']['remote']['code_root']
        root = TrainConfig['path']['remote']['code_root']
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Set environment variables to force synchronization and get an accurate stack
        # os.environ["TORCH_USE_CUDA_DSA"] = "1" # Enable CUDA dynamic concurrent allocation to reduce memory fragmentation

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:32"

    else:
        print("Unknown operating system, unable to set path")
        sys.exit(1)

    sys.path.append(code_path)
    return code_path, root
