# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2024/4/14 20:04
@Author  : H-Tenets
@File    : TrainSample.py
@Software: PyCharm 
@Git-v   : 
"""
import os
from SetRunPath import setRunPath

runPath, root = setRunPath()
current_file_name = os.path.basename(__file__)
print("{}: Code execution path: {}, Root path: {}".format(current_file_name, runPath, root))

from utils.DeformedFringe import generate_deformed_stripes_single
from utils.myutils import *
import torch.nn as nn


def TwoStream_trainSample(sample, model, optimizer, criterion, labeled_bs=8):
    model.train()
    sample_cuda = tocuda(sample)
    image_label = sample_cuda["image_T32"]
    depthStandard = sample_cuda["depth"]
    phaStandard = sample_cuda["pha_diff"]

    cm2mm = 10
    d = sample_cuda["d"][0].item()
    l = sample_cuda["l"][0].item()
    p = sample_cuda["p"][0].item()
    T = sample_cuda["T32"][0].item()
    depthStandard *= cm2mm

    PhaDiffPrediction = model.forward(image_label)
    depthPrediction = pha2depth(PhaDiffPrediction, d, l, p)
    PhaDiff2ImagePre = generate_deformed_stripes_single(PhaDiffPrediction, T)
    loss = criterion(depthPrediction, depthStandard, PhaDiff2ImagePre, image_label, labeled_bs)

    MAELoss = nn.L1Loss(reduction='mean')
    MSELoss = nn.MSELoss()
    loss_Depth_labeled_MAE = MAELoss(depthPrediction[:labeled_bs], depthStandard[:labeled_bs])
    loss_Depth_unlabeled_MAE = MAELoss(depthPrediction[labeled_bs:] * cm2mm, depthStandard[labeled_bs:])
    loss_Pha_MAE = MAELoss(PhaDiffPrediction, phaStandard)
    loss_Depth_MAE = MAELoss(depthPrediction, depthStandard)
    loss_Depth_RMSE = torch.sqrt(MSELoss(depthPrediction, depthStandard))

    scalar_outputs = {
        "loss": loss,
        "loss_Depth_labeled_MAE": loss_Depth_labeled_MAE,
        "loss_Depth_unlabeled_MAE": loss_Depth_unlabeled_MAE,
        "loss_Pha_MAE": loss_Pha_MAE,
        "loss_Depth_MAE": loss_Depth_MAE,
        "loss_Depth_RMSE": loss_Depth_RMSE,
    }
    image_outputs = {
        "PhaDiff2ImagePre": PhaDiff2ImagePre,
        "imageStandard": image_label,
        "PhaDiffPrediction": PhaDiffPrediction,
        "phaStandard": phaStandard,
    }

    return loss, tensor2float(scalar_outputs), image_outputs
