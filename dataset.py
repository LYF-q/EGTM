import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, random_split
import custom_transforms as tr

from osgeo import gdal
from PIL import Image
import numpy as np
from torch.utils.data.dataset import random_split

from osgeo import gdal, gdal_array, gdalconst
import random
import cv2


class Terrace_Loader(Dataset):

    def __init__(self, source_path):
        i = 0
        self.source_path = source_path

        self.labels = os.path.join(source_path, 'label_train_val.txt')
        self.imgs = os.path.join(source_path, 'image_train_val.txt')

        self.imgs_path = []
        self.labels_path = []

        with open(self.imgs, 'r') as file:
            lines = file.readlines()
            for line in lines:
                string = line.strip()
                self.imgs_path.append(string)

        with open(self.labels, 'r') as file:
            lines = file.readlines()
            for line in lines:
                string = line.strip()
                self.labels_path.append(string)

    def _make_img_gt_point_pair(self, index):
        # retrain
        _img = gdal.Open(self.imgs_path[index])
        _target = gdal.Open(self.labels_path[index])

        return _img, _target

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        labels_path =self.labels_path[index]
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        return self.transform_tr(sample)

    def __len__(self):
        return len(self.imgs_path)

    @staticmethod
    def transform_tr(sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.RandomColorJitterMultiBand(),
            tr.RandomDoubleNoise(),
            tr.RandomCloud(),
            tr.RandomErasing(),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def transform_val(sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()])

        return composed_transforms(sample)
