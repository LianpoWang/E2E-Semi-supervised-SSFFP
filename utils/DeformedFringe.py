# -*- coding: utf-8 -*-

"""
@Project : UnsupervisedDiffusionModel
@Time    : 2024/3/11 19:56
@Author  : H-Tenets
@File    : DeformedFringe.py
@Software: PyCharm 
"""

import torch


def generate_deformed_stripes_single(phase_difference, period):
    """
    Generate a single deformed fringe pattern based on phase difference and fringe period.

    Args:
        phase_difference (torch.Tensor): Phase difference map of shape (B, C, H, W)
        period (float): Fringe period (in pixels)

    Returns:
        torch.Tensor: Normalized deformed stripe pattern of shape (H, W)
    """

    _, _, height, width = phase_difference.shape

    frequency = 1.0 / period

    # Compute deformation parameters
    A = 115
    B = 100

    # Create horizontal coordinate grid (j indices), reversed to start deformation from the right side for vertical stripes
    j_indices = torch.arange(width-1,-1,-1).float().unsqueeze(0).repeat(height, 1).cuda()
    deformed_stripes = A + B * torch.cos(2 * torch.pi * j_indices * frequency + phase_difference+torch.pi/16)

    deformed_stripes = (deformed_stripes - deformed_stripes.min()) / (deformed_stripes.max() - deformed_stripes.min())

    return deformed_stripes


