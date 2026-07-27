
import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from osgeo import gdal
from torch.utils.data.dataset import random_split
from osgeo import gdal, gdal_array, gdalconst
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.e = 1e-38

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        mask = mask.ReadAsArray().astype(np.float32)
        img = img.ReadAsArray().astype(np.float32)

        return {'image': img,
                'label': mask}

class ToTensor(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        return {'image': img,
                'label': mask}

class RandomColorJitterMultiBand(object):
    def __init__(self, brightness=0.15, contrast=0.15, prob=0.5, bands_to_jitter=4):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob
        self.bands_to_jitter = bands_to_jitter

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < self.prob:
            jittered = img.astype(np.float32).copy()
            C = img.shape[0]
            for c in range(min(self.bands_to_jitter, C)):
                band = img[c].astype(np.float32)

                b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                band = band * b_factor

                c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                mean_val = band.mean()
                band = (band - mean_val) * c_factor + mean_val

                jittered[c] = band

            img = jittered

        return {'image': img.astype(np.float32), 'label': mask}

class RandomDoubleNoise(object):
    def __init__(self, gauss_sigma=0.01, saltpepper_prob=0.01, bands_to_noise=4, prob=0.2):

        self.gauss_sigma = gauss_sigma
        self.saltpepper_prob = saltpepper_prob
        self.bands_to_noise = bands_to_noise
        self.prob = prob

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < self.prob:
            noisy = img.copy().astype(np.float32)
            C, H, W = img.shape

            for c in range(min(self.bands_to_noise, C)):
                band = noisy[c]
                band_range = band.max() - band.min()
                sigma = self.gauss_sigma * band_range
                noise = np.random.normal(0, sigma, size=band.shape)
                noisy[c] = band + noise

            rand_matrix = np.random.rand(H, W)
            salt_mask = rand_matrix < (self.saltpepper_prob / 2)
            pepper_mask = (rand_matrix >= (self.saltpepper_prob / 2)) & (rand_matrix < self.saltpepper_prob)

            for c in range(min(self.bands_to_noise, C)):
                band = noisy[c]
                band[salt_mask] = band.max()
                band[pepper_mask] = band.min()
                noisy[c] = band

            img = noisy

        return {'image': img.astype(np.float32), 'label': mask}


class RandomCloud(object):
    def __init__(self, cloud_prob=0.1, min_alpha=0.3, max_alpha=0.6, cloud_size=0.05):

        self.cloud_prob = cloud_prob
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.cloud_size = cloud_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < self.cloud_prob:
            C, H, W = img.shape

            noise = np.random.randn(H, W).astype(np.float32)
            ksize = int(min(H, W) * self.cloud_size)
            if ksize % 2 == 0:
                ksize += 1
            cloud_mask = cv2.GaussianBlur(noise, (ksize, ksize), 0)
            cloud_mask = (cloud_mask - cloud_mask.min()) / (cloud_mask.max() - cloud_mask.min())
            alpha = random.uniform(self.min_alpha, self.max_alpha)
            img_min, img_max = img[:4].min(), img[:4].max()
            for c in range(min(4, C)):
                img[c] = img[c] * (1 - alpha * cloud_mask) + img_max * (alpha * cloud_mask)

        return {'image': img.astype(np.float32), 'label': mask}


class RandomErasing(object):
    def __init__(self,
                 prob=0.05,
                 max_area_ratio=0.05,
                 min_area_ratio=0.02,
                 num_blocks=(1, 2),
                 bands_to_occlude=(0, 1, 2, 3),
                 slope_index=5,
                 occlude_slope=False,
                 fill_mode='mean',
                 fill_value=0):
        self.prob = prob
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.num_blocks = num_blocks
        self.bands_to_occlude = set(bands_to_occlude)
        self.slope_index = slope_index
        self.occlude_slope = occlude_slope
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def _get_fill_patch(self, patch_shape, img_band, band_mean):
        if self.fill_mode == 'constant':
            return np.full(patch_shape, self.fill_value, dtype=img_band.dtype)
        elif self.fill_mode == 'mean':
            return np.full(patch_shape, band_mean, dtype=img_band.dtype)
        elif self.fill_mode == 'noise':
            std = np.std(img_band)
            return np.random.normal(loc=band_mean, scale=std + 1e-6, size=patch_shape).astype(img_band.dtype)
        else:
            return np.full(patch_shape, 0, dtype=img_band.dtype)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() >= self.prob:
            return {'image': img, 'label': mask}

        C, H, W = img.shape
        total_area = H * W
        n_blocks = random.randint(self.num_blocks[0], self.num_blocks[1])

        target_bands = set(self.bands_to_occlude)
        if self.occlude_slope and (0 <= self.slope_index < C):
            target_bands.add(self.slope_index)

        band_means = [np.mean(img[c]) for c in range(C)]

        for _ in range(n_blocks):
            area_ratio = random.uniform(self.min_area_ratio, self.max_area_ratio)
            occ_area = int(area_ratio * total_area)

            target_h = max(1, int(np.sqrt(occ_area * (H / W))))
            target_w = max(1, int(np.sqrt(occ_area * (W / H))))

            stretch_h = random.uniform(0.6, 1.4)
            stretch_w = random.uniform(0.6, 1.4)
            occ_h = np.clip(int(target_h * stretch_h), 1, H)
            occ_w = np.clip(int(target_w * stretch_w), 1, W)

            top = random.randint(0, H - occ_h)
            left = random.randint(0, W - occ_w)

            for c in target_bands:
                band_mean = band_means[c]
                patch = self._get_fill_patch((occ_h, occ_w), img[c], band_mean)
                img[c, top:top+occ_h, left:left+occ_w] = patch
            mask[top:top + occ_h, left:left + occ_w] = 0

        return {'image': img, 'label': mask}
