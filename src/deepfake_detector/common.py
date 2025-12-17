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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, custom_labels=None):

        if custom_labels is not None:
            self.labels = custom_labels
        else:
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
