# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2024/4/15 12:16
@Author  : H-Tenets
@File    : ValSample.py
@Software: PyCharm 
@Git-v   : 4.0
"""
import os

from SetRunPath import setRunPath
import torch.nn as nn

runPath, root = setRunPath()
current_file_name = os.path.basename(__file__)
print("{}: Code execution path: {}, Root path: {}".format(current_file_name, runPath, root))

from utils.DeformedFringe import generate_deformed_stripes_single
from utils.myutils import *


@make_nograd_func
def TwoStream_valSample(sample, model):
    model.eval()
    sample_cuda = tocuda(sample)
    image_label = sample_cuda["image_T32"]
    depthStandard = sample_cuda["depth"]
    phaStandard = sample_cuda["pha_diff"]

    d = sample_cuda["d"][0].item()
    l = sample_cuda["l"][0].item()
    p = sample_cuda["p"][0].item()
    T = sample_cuda["T32"][0].item()
    cm2mm = 10
    depthStandard *= cm2mm

    with torch.no_grad():

        PhaDiffPrediction = model.forward(image_label)
        depthPrediction = pha2depth(PhaDiffPrediction, d, l, p)
        PhaDiff2ImagePre = generate_deformed_stripes_single(PhaDiffPrediction, T)

        MAELoss = nn.L1Loss(reduction='mean')
        MSELoss = nn.MSELoss()
        loss_Pha_MAE = MAELoss(PhaDiffPrediction, phaStandard)
        loss_Depth_MAE = MAELoss(depthPrediction, depthStandard)
        loss_Depth_RMSE = torch.sqrt(MSELoss(depthPrediction, depthStandard))

    scalar_outputs = {
        "loss": loss_Depth_MAE,
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

    return tensor2float(loss_Depth_MAE), tensor2float(scalar_outputs), image_outputs
