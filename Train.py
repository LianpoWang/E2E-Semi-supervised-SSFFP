# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2025/10/25 11:10
@Author  : H-Tenets
@File    : Train_TwoStream.py
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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.benchmark = True  # Optimize for speed (non-deterministic)
    torch.backends.cudnn.deterministic = False


set_seed(425)

from SetRunPath import setRunPath

runPath, root = setRunPath()
current_file_name = os.path.basename(__file__)
print("{}：代码运行路径为:{}，根路径为：{}".format(current_file_name, runPath, root))

from utils.myutils import *
from datasets.Dataset import TwoStreamBatchSampler
from torch.utils.data import DataLoader
import importlib

# Dynamically load modules based on config
TrainConfig = toml.load("TrainConfig.toml")
currentModel = TrainConfig['models']['sorts']['current_model']

if currentModel == 'TwoStream':
    Model = TrainConfig['models']['TwoStream']
else:
    # 输入错误提示
    print("模型不存在")
    sys.exit(1)  # 终止程序，并返回非零值表示错误

imports = Model['imports']

# Parse module paths and class names
dataset_module_path, dataset_class_name = imports['dataset_module'].rsplit('.', maxsplit=1)
model_module_path, model_class_name = imports['model_module'].rsplit('.', maxsplit=1)
loss_module_path, loss_class_name = imports['loss_module'].rsplit('.', maxsplit=1)
trainSample_module_path, trainSample_class_name = imports['train_sample_module'].rsplit('.', maxsplit=1)
valSample_module_path, valSample_class_name = imports['val_sample_module'].rsplit('.', maxsplit=1)

# Dynamically import modules
dataset_module = importlib.import_module(dataset_module_path)
model_module = importlib.import_module(model_module_path)
loss_module = importlib.import_module(loss_module_path)
trainSample_module = importlib.import_module(trainSample_module_path)
valSample_module = importlib.import_module(valSample_module_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select local/remote mode
parser_path = argparse.ArgumentParser(description='The Path Selection For The Model To Run')
parser_path.add_argument('--path_mode', default='remote', help='local or remote', choices=['local', 'remote'])
args_path = parser_path.parse_args()

parser = argparse.ArgumentParser(description='Parameter Settings For The Model Run')

if args_path.path_mode == 'local':
    Path = TrainConfig['path']['local']
elif args_path.path_mode == 'remote':
    Path = TrainConfig['path']['remote']
else:
    print("输入错误：mode参数只能是'local'、'remote'")
    sys.exit(1)

model_name = Model['name']
# dataset_name = Model['dataset']
is_resume = Model['resume']
extra_path = Model['extra']

parser.add_argument('--data_root', default=Path['data_root'])
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])

# log_save 最终路径由 log_save_dir + current_model.name + checkpoints/checkpoints_beat + extra_path 组成
parser.add_argument('--logdir', default=os.path.join(Path['log_save_dir'], model_name, "checkpoints/", extra_path),
                    help='the directory to save checkpoints/logs')
parser.add_argument('--savedir',
                    default=os.path.join(Path['log_save_dir'], model_name, "checkpoints_best/", extra_path),
                    help='the directory to save checkpoints_best/logs')

parser.add_argument('--epochs', type=int, default=Path['epochs'], help='number of epochs to train')  # 600
parser.add_argument('--batch_size', type=int, default=Path['batch_size'], help='train batch size')  # 12
parser.add_argument('--labeled_bs', type=int, default=1, help='train batch size')
parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation steps for large batch size')  # 4

parser.add_argument('--best_loss_train', type=int, default=Path['best_loss_train'])  # 1000
parser.add_argument('--best_loss_val', type=int, default=Path['best_loss_val'])  # 1000
parser.add_argument('--summary_freq', type=int, default=Path['summary_freq'], help='print and summary frequency')  # 20
parser.add_argument('--save_freq', type=int, default=Path['save_freq'], help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=Path['seed'], metavar='S', help='random seed')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="29,69,99:10",
                    help='epoch ids to downscale lr and the downscale rate')

parser.add_argument('--wd', type=float, default=0.0000, help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0)

parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--valpath', help='val datapath')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--loadpth', default=None, help='load a specific checkpoint')

parser.add_argument('--resume', action='store_true', help='continue to train the model')

# Construct args programmatically based on config
construct_args = ['--mode=train']
if is_resume:
    construct_args.append('--resume')
args = parser.parse_args(construct_args)

if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
    assert args.loadpth is None
if args.valpath is None:
    args.valpath = args.trainpath

if args.mode == "train":
    if not os.path.isdir(args.logdir):
        # os.mkdir(args.logdir)
        os.makedirs(args.logdir)
    if not os.path.isdir(args.savedir):
        # os.mkdir(args.savedir)
        os.makedirs(args.savedir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print_args(args)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


dentalphase = getattr(dataset_module, dataset_class_name)

dataset_path = os.path.join(args.data_root, "NewDatasetImageDepthT32T64_0-1.npz")  #
# dataset_path = os.path.join(args.data_root, "RealDatasetImagePhase.npz")  #


unlabeledTrain_dataset = dentalphase(dataset_path, "unlabeledTrain")
labeledTrain_dataset = dentalphase(dataset_path, "labeledTrain")
allTrain_dataset = dentalphase(dataset_path, "allTrain")
val_dataset = dentalphase(dataset_path, "val")

labeled_idxs = list(range(0, len(labeledTrain_dataset)))
unlabeled_idxs = list(range(len(labeledTrain_dataset), len(allTrain_dataset)))

# ----单流采样器----
# TrainImgLoader = DataLoader(allTrain_dataset, args.batch_size, shuffle=True, num_workers=TrainConfig['thread'],
#                             worker_init_fn=worker_init_fn)
# TrainImgLoader = DataLoader(unlabeledTrain_dataset, args.batch_size, shuffle=True, num_workers=TrainConfig['thread'], worker_init_fn=worker_init_fn)
# TrainImgLoader = DataLoader(labeledTrain_dataset, args.batch_size, shuffle=True, num_workers=TrainConfig['thread'], worker_init_fn=worker_init_fn)

# ----双流采样器----
batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
TrainImgLoader = DataLoader(allTrain_dataset, batch_sampler=batch_sampler, num_workers=TrainConfig['thread'],
                            worker_init_fn=worker_init_fn)

# ----finally----
ValImgLoader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=TrainConfig['thread'])

# 配置文件定义损失函数
criterion = getattr(loss_module, loss_class_name)().cuda()

# NEW 4.0 新增反射机制
model = getattr(model_module, model_class_name)()
model.cuda()
model = torch.nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt and not args.loadpth):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models,
                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.logdir, saved_models[-1])

    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    args.start_epoch = state_dict['epoch'] + 1
    args.best_loss_train = state_dict['best_loss_train']
    args.best_loss_val = state_dict['best_loss_val']

elif args.loadckpt and not args.loadpth:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    loadckpt = os.path.join(args.logdir, args.loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

elif args.loadpth and not args.loadckpt:
    # load .pth file specified by args.pth
    print("loading model {}".format(args.loadpth))
    loadpth = os.path.join(args.savedir, args.loadpth)
    state_dict = torch.load(loadpth)
    model.load_state_dict(state_dict)

print("start at epoch {} to {}".format(args.start_epoch, args.start_epoch + args.epochs))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def train():
    start_epoch = args.start_epoch
    best_loss_train = args.best_loss_train
    best_loss_val = args.best_loss_val

    start_time = 0
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    accumulation_steps = args.accumulation_steps

    for epoch_idx in range(start_epoch, start_epoch + args.epochs):
        trainProgress_bar = tqdm(enumerate(TrainImgLoader), total=len(TrainImgLoader),
                                 desc="TRAIN {}/{} Process".format(epoch_idx, start_epoch + args.epochs))
        avg_train_scalars = DictAverageMeter()
        optimizer.zero_grad()
        for batch_idx, sample in trainProgress_bar:
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            train_sample = getattr(trainSample_module, trainSample_class_name, 'not find')
            batch_loss, scalar_outputs, image_outputs = train_sample(sample, model, optimizer, criterion,
                                                                     args.labeled_bs)

            batch_loss = batch_loss / accumulation_steps
            batch_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(TrainImgLoader):
                optimizer.step()
                optimizer.zero_grad()

            current_batch_loss = batch_loss.item() * accumulation_steps

            scalar_outputs['loss'] = current_batch_loss

            if do_summary:
                save_images(logger, 'train_global', image_outputs, global_step)

            avg_train_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs

        epoch_train_mean_losses = avg_train_scalars.mean()
        train_mean_loss = epoch_train_mean_losses.get('loss', None)

        save_scalars(logger, 'train_epoch', epoch_train_mean_losses, epoch_idx)
        save_scalars_to_csv(os.path.join(args.savedir, 'train_loss.csv'), epoch_idx, epoch_train_mean_losses,
                            write_header=True)

        print(f'TRAIN {epoch_idx}       LOSS:', end=' ')
        for loss_name, loss_value in epoch_train_mean_losses.items():
            print(f'{loss_name}={loss_value:.4f}', end=', ')
        print(f'time = {time.time() - start_time:.2f}, epoch_lr: {lr_scheduler.get_last_lr()}')

        model_best_train_pth_name = f'{args.savedir}/model_best_train.pth'
        if train_mean_loss < best_loss_train:
            best_loss_train = train_mean_loss
            torch.save(model.state_dict(), model_best_train_pth_name)
            print(
                f'save best TRAIN model: {model_best_train_pth_name} (epoch: {epoch_idx}),train_loss: {train_mean_loss:.4f}')
            with open(args.savedir + "/best_train.txt", "w") as myfile:
                myfile.write("Best train epoch is %d, with train-Loss= %.4f" % (epoch_idx, train_mean_loss))

        # validate
        avg_val_scalars = DictAverageMeter()
        valProgress_bar = tqdm(enumerate(ValImgLoader), total=len(ValImgLoader),
                               desc="VAL   {}/{} Process".format(epoch_idx, start_epoch + args.epochs))
        for batch_idx, sample in valProgress_bar:
            start_time = time.time()
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            val_sample = getattr(valSample_module, valSample_class_name)
            val_loss, scalar_outputs, image_outputs = val_sample(sample, model)
            if do_summary:
                save_images(logger, 'val_global', image_outputs, global_step)
            avg_val_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs

        save_scalars(logger, 'val_epoch', avg_val_scalars.mean(), epoch_idx)

        epoch_val_mean_losses = avg_val_scalars.mean()
        val_mean_loss = epoch_val_mean_losses.get('loss', None)
        save_scalars_to_csv(os.path.join(args.savedir, 'val_loss.csv'), epoch_idx, epoch_val_mean_losses,
                            write_header=True)

        print(f'VAL   {epoch_idx}       LOSS:', end=' ')
        for loss_name, loss_value in epoch_val_mean_losses.items():
            print(f'{loss_name}={loss_value:.4f}', end=', ')
        print(f'time = {time.time() - start_time:.2f}, epoch_lr: {lr_scheduler.get_last_lr()}')

        model_best_val_pth_name = f'{args.savedir}/model_best_val.pth'
        if val_mean_loss < best_loss_val:
            best_loss_val = val_mean_loss
            torch.save(model.state_dict(), model_best_val_pth_name)
            print(f'save best VAL model: {model_best_val_pth_name} (epoch: {epoch_idx}, val_loss: {val_mean_loss:.4f})')
            with open(args.savedir + "/best_val.txt", "w") as myfile:
                myfile.write("Best val epoch is %d, with val-Loss= %.4f" % (epoch_idx, val_mean_loss))

        if (epoch_idx + 1) % args.save_freq == 0:
            try:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss_train': best_loss_train,
                    'best_loss_val': best_loss_val,
                }, "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            except RuntimeError as e:
                print("Caught a RuntimeError while saving the model:", e)
            except Exception as e:
                print("Caught an exception:", e)
        lr_scheduler.step()


if __name__ == '__main__':
    if args.mode == "train":
        train()
