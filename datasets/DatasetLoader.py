# -*- coding: utf-8 -*-
"""
@Project : GraduateStudentAllNet
@Time    : 2024/12/26 09:57
@Author  : H-Tenets
@File    : DatasetLoader.py
@Software: PyCharm 
@Git-v   : 
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor

from utils.myutils import depth2pha

np.random.seed(313)  # Ensure the dataset split is consistent across runs


class NPZDataset_SimulationStripes(Dataset):
    """
    Save images and depth information from the dataset into a .npz file.
    Other information such as phase or masks will be computed during data loading.
    """
    _data_loaded = False
    _images_T32 = None
    _images_T64 = None
    _depths = None
    _pha_diffs = None
    _ds = None
    _l = None
    _p = None
    _train_indices = None
    _val_indices = None
    _test_indices = None
    _single_dl_nums = 1200  # Number of samples per dl
    _data_nums = None  # Total number of samples

    def __init__(self, npz_file, subset="train", isAugment=True):
        """
        Initialize the dataset with support for train, validation, and test splits.
        Args:
            npz_file (str): Path to the .npz file
            subset (str): Type of dataset split: "allTrain", "labeledTrain", "unlabeledTrain", "val", or "test"
        """
        if not NPZDataset_SimulationStripes._data_loaded:
            self.load_data(npz_file, isAugment)

        # Select indices based on the specified subset
        if subset == "allTrain":
            self.indices = NPZDataset_SimulationStripes._allTrain_indices
        elif subset == "labeledTrain":
            self.indices = NPZDataset_SimulationStripes._labeledTrain_indices
        elif subset == "unlabeledTrain":
            self.indices = NPZDataset_SimulationStripes._unlabeledTrain_indices
        elif subset == "val":
            self.indices = NPZDataset_SimulationStripes._val_indices
        elif subset == "test":
            self.indices = NPZDataset_SimulationStripes._test_indices
        else:
            raise ValueError("subset must be 'train', 'val' or 'test'")

    @classmethod
    def load_data(cls, npz_file, isAugment):
        """
        Load data from an NPZ file and perform data augmentation if enabled.
        Args:
            npz_file (str): Path to the .npz file
        """
        data = np.load(npz_file)
        cls._images_T32 = data['images_T32']
        # cls._images_T64 = data['images_T64']
        cls._depths = data['depths']
        cls._ds = np.array([60] * cls._single_dl_nums + [70] * cls._single_dl_nums + [80] * cls._single_dl_nums)
        cls._l = 20
        cls._p = 0.875
        cls._data_nums = cls._images_T32.shape[0]
        # cls._data_nums = cls._images_T64.shape[0]
        if isAugment:
            total_samples = cls._data_nums * 2
        else:
            total_samples = cls._data_nums

        del cls._images_T64  # Free memory

        labeledTrain_ratio = 0.4
        unlabeledTrain_ratio = 0.4
        val_ratio = 0.1
        test_ratio = 0.1

        # Prepare indices
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # Compute split boundaries
        allTrain_end = int((labeledTrain_ratio + unlabeledTrain_ratio) * total_samples)
        labeledTrain_end = int(labeledTrain_ratio * total_samples)
        unlabeledTrain_end = int((labeledTrain_ratio + unlabeledTrain_ratio) * total_samples)
        val_end = int((labeledTrain_ratio + unlabeledTrain_ratio + val_ratio) * total_samples)
        test_end = int((labeledTrain_ratio + unlabeledTrain_ratio + val_ratio + test_ratio) * total_samples)

        cls._allTrain_indices = indices[:allTrain_end]
        cls._labeledTrain_indices = indices[:labeledTrain_end]
        cls._unlabeledTrain_indices = indices[labeledTrain_end:unlabeledTrain_end]
        cls._val_indices = indices[unlabeledTrain_end:val_end]
        cls._test_indices = indices[val_end:]
        cls._data_loaded = True  # Set the flag to True to indicate data has been loaded

    @classmethod
    def augment_data(cls, data):
        train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
        ])
        augmented_img = np.array(train_transform(Image.fromarray(data)))
        return augmented_img

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if real_idx < self._data_nums:  # Original data
            image_T32 = ToTensor()(self._images_T32[real_idx])
            # image_T64=ToTensor()(self._images_T64[real_idx])
            depth = ToTensor()(self._depths[real_idx])
            pha_diff = ToTensor()(
                depth2pha(self._depths[real_idx], self._ds[real_idx], self._l, self._p))
        else:  # Augmented data
            real_idx = real_idx - self._data_nums  # 增广数据的索引需要减去原始数据的数量
            image_T32 = ToTensor()(self.augment_data(self._images_T32[real_idx]))
            # image_T64=ToTensor()(self.augment_data(self._images_T64[real_idx]))
            depth = ToTensor()(self.augment_data(self._depths[real_idx]))
            pha_diff = ToTensor()(
                self.augment_data(depth2pha(self._depths[real_idx], self._ds[real_idx], self._l, self._p)))
        return {
            "image_T32": image_T32,
            # "image_T64": image_T64,
            "depth": depth,
            "pha_diff": pha_diff,  # Phase difference computed during training
            "d": self._ds[real_idx],
            "l": self._l,
            "p": self._p,  # Width parameter for T32; T64 would require multiplying by 2
            "T32": 32,
            # "T64": 64,
            "imgfilename": str(real_idx),
        }
