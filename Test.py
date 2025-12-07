# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2025/1/3 10:35
@Author  : H-Tenets
@File    : Test_npy.py
@Software: PyCharm 
@Git-v   :
"""

import argparse
import datetime
import os
import random
import sys
import time

import toml
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Disable cuDNN benchmarking to ensure reproducibility
    torch.backends.cudnn.benchmark = False
    # Ensure deterministic behavior in cuDNN operations
    torch.backends.cudnn.deterministic = True

# Set random seed at the beginning of the script
set_seed(425)  # Example seed; can be adjusted as needed
from SetRunPath import setRunPath

runPath, root = setRunPath()  # Get the current running file path
current_file_name = os.path.basename(__file__)
print("{}: Code execution path: {}, Root path: {}".format(current_file_name, runPath, root))

from utils.myutils import *

# Load test configuration from TOML file
TestConfig = toml.load("TestConfig.toml")
currentModel = TestConfig['models']['sorts']['current_model']

if currentModel == 'TwoStream':
    Model = TestConfig['models']['TwoStream']
else:
    print("Error: Model not found.")
    sys.exit(1)  # Exit with error code

imports = Model['imports']


# Extract module paths and class names for dataset, model, and test sampler
dataset_module_path, dataset_class_name = imports['dataset_module'].rsplit('.', maxsplit=1)
model_module_path, model_class_name = imports['model_module'].rsplit('.', maxsplit=1)
testSample_module_path, testSample_class_name = imports['test_sample_module'].rsplit('.', maxsplit=1)

# Dynamically import modules using importlib
dataset_module = importlib.import_module(dataset_module_path)
model_module = importlib.import_module(model_module_path)
testSample_module = importlib.import_module(testSample_module_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First ArgumentParser: select path mode (local or remote)
parser_path = argparse.ArgumentParser(description='The Path Selection For The Model To Run')
parser_path.add_argument('--path_mode', default='remote', help='local or remote', choices=['local', 'remote'])
args_path = parser_path.parse_args()

# Second ArgumentParser: configure test parameters
parser = argparse.ArgumentParser(description='Parameter Settings For The Model Run')

if args_path.path_mode == 'local':
    Path = TestConfig['path']['local']
elif args_path.path_mode == 'remote':
    Path = TestConfig['path']['remote']
else:
    print("Error: --path_mode must be either 'local' or 'remote'")
    sys.exit(1)

# 加载模型的名字和数据集
model_name = Model['name']
model_path = Model['model_path']
test_outputs = Model['test']

parser.add_argument('--data_root', default=Path['data_root'])
parser.add_argument('--mode', default='test', help='test', choices=['test', 'profile'])

parser.add_argument('--ckpt_load_dir',
                    default=os.path.join(Path['modelStored_root'], model_name, "checkpoints/", model_path),
                    help='the directory to save model of train')

parser.add_argument('--pth_load_dir',
                    default=os.path.join(Path['modelStored_root'], model_name, "checkpoints_best/", model_path),
                    help='the directory to save model of train')

parser.add_argument('--test_logdir',
                    default=os.path.join(Path['test_log_save_root'], model_name, "test/logs", test_outputs),
                    help='the directory to save logs of test')
parser.add_argument('--test_savedir',
                    default=os.path.join(Path['test_log_save_root'], model_name, "test/results/", test_outputs),
                    help='the directory to save results of test')

parser.add_argument('--batch_size', type=int, default=Path['batch_size'], help='test batch size')
parser.add_argument('--summary_freq', type=int, default=Path['summary_freq'], help='print and summary frequency')  # 20
parser.add_argument('--save_freq', type=int, default=Path['save_freq'], help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=Path['seed'], metavar='S', help='random seed')

parser.add_argument('--loss_test_list', type=list, default=[])
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')

construct_args = ['--mode=test']
args = parser.parse_args(construct_args)

# create logger for mode "train" and "testall"
if args.mode == "test":
    if not os.path.isdir(args.test_logdir):
        # os.mkdir(args.test_logdir)
        os.makedirs(args.test_logdir)
    if not os.path.isdir(args.test_savedir):
        # os.mkdir(args.test_savedir)
        os.makedirs(args.test_savedir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.test_logdir)

# Print parsed arguments (assumes print_args is defined in myutils)
print_args(args)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

# Dynamically instantiate dataset using reflection
dentalphase = getattr(dataset_module, dataset_class_name)


dataset_path = os.path.join(args.data_root, "NewDatasetImageDepthT32T64_0-1.npz")  #
# dataset_path = os.path.join(args.data_root, "RealDatasetImagePhase.npz")  #


test_dataset = dentalphase(dataset_path, "test")

TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=TestConfig['thread'],worker_init_fn=worker_init_fn)

model = getattr(model_module, model_class_name)()
model = torch.nn.DataParallel(model)
model.cuda()

# load parameters
if args.mode == "test" and not args.loadckpt:
    # Option 1: Load latest .ckpt (commented out)
    # saved_models = [fn for fn in os.listdir(args.ckpt_load_dir) if fn.endswith(".ckpt")]
    # saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # # use the latest checkpoint file
    # loadckpt = os.path.join(args.ckpt_load_dir, saved_models[-1])
    # print("resuming", loadckpt)
    # state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'])

    # Option 2: Load best .ckpt model
    # saved_model = "model_000199.ckpt"
    # loadckpt = os.path.join(args.ckpt_load_dir, saved_model)
    # print("resuming", loadckpt)
    # state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'])

    # Option 2: Load best .pth model
    loadpth = os.path.join(args.pth_load_dir, 'model_best_val.pth')
    state_dict = torch.load(loadpth)
    model.load_state_dict(state_dict)


elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    loadckpt = os.path.join(args.ckpt_load_dir, args.loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function

def test():

    testProgress_bar = tqdm(enumerate(TestImgLoader), total=len(TestImgLoader), desc="TEST Process")
    avg_test_scalars = DictAverageMeter()
    MAE_loss_list = []
    loss_avgtest_list = []

    for batch_idx, sample in testProgress_bar:
        start_time = time.time()

        global_step = len(TestImgLoader) + batch_idx
        do_summary = global_step % args.summary_freq == 0

        test_sample = getattr(testSample_module, testSample_class_name)
        MAE_loss, scalar_outputs, image_outputs = test_sample(sample, model,args.test_savedir)


        if do_summary:
            save_scalars(logger, 'test_global', scalar_outputs, global_step)
            save_images(logger, 'test_global', image_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)

        print(f' Batch Iter {batch_idx}/{len(TestImgLoader)}  LOSS:', end=' ')
        for loss_name, loss_value in scalar_outputs.items():
            print(f'{loss_name}={loss_value:.4f}', end=', ')
        print(f'time = {time.time() - start_time:.3f}')

        del scalar_outputs, image_outputs

        loss_avgtest_list.append(avg_test_scalars.mean())
        MAE_loss_list.append(MAE_loss)

    print(f"model_path:{model_path}", end=', ')
    for loss_name, loss_value in avg_test_scalars.mean().items():
        print(f'{loss_name}={loss_value:.4f}', end=', ')

if __name__ == '__main__':
    test()