# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2024/10/22 下午3:49
@Author  : H-Tenets
@File    : Dataset.py
@Software: PyCharm
@article{wang2024mamba,
  title={Mamba-unet: Unet-like pure visual mamba for medical image segmentation},
  author={Wang, Ziyang and Zheng, Jian-Qing and Zhang, Yichi and Cui, Ge and Li, Lei},
  journal={arXiv preprint arXiv:2402.05079},
  year={2024}
  https://github.com/ziyangwang007/Mamba-UNet/blob/main/code/augmentations/ctaugment.py
}

"""
import itertools

import numpy as np
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
        primary_indices (list):
        secondary_indices (list):
        batch_size (int):
        secondary_batch_size (int):
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
