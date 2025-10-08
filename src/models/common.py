from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.labels = [label for _, label in dataset]
        self.label_to_indices = defaultdict(list)

        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.batch_size = batch_size
        self.half_batch = batch_size // 2

        # Shuffle indices
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])
        self.min_class_len = min(len(self.label_to_indices[0]), len(self.label_to_indices[1]))
        self.num_batches = self.min_class_len * 2 // self.batch_size

    def __iter__(self):
        indices_0 = self.label_to_indices[0].copy()
        indices_1 = self.label_to_indices[1].copy()
        random.shuffle(indices_0)
        random.shuffle(indices_1)

        for i in range(self.num_batches):
            batch = indices_0[i * self.half_batch:(i + 1) * self.half_batch] + \
                    indices_1[i * self.half_batch:(i + 1) * self.half_batch]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class FFTImageDataset(Dataset):
    def __init__(self, image_folder, transform_rgb=None):
        self.dataset = image_folder
        self.transform_rgb = transform_rgb

    def __len__(self):
        return len(self.dataset)

    def compute_fft(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        magnitude = torch.tensor(magnitude).unsqueeze(0).float()
        return magnitude

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_np = np.array(image)

        fft_input = self.compute_fft(image_np)

        if self.transform_rgb:
            rgb_input = self.transform_rgb(image)
        else:
            rgb_input = to_tensor(image)

        return {
            "rgb_input": rgb_input,
            "fft_input": fft_input
        }, label


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform_rgb=None):
        self.dataset = image_folder
        self.transform_rgb = transform_rgb

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform_rgb:
            rgb_input = self.transform_rgb(image)
        else:
            rgb_input = to_tensor(image)

        return {
            "rgb_input": rgb_input
        }, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = Fn.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean()
