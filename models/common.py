from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
import random
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import io

class AdvancedAugment:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = self.jpeg_compress(img)
        if random.random() < self.prob:
            img = self.gaussian_blur(img)
        if random.random() < self.prob:
            img = self.color_jitter(img)
        if random.random() < self.prob:
            img = self.add_gaussian_noise(img)
        return img

    def jpeg_compress(self, img, quality_range=(30, 70)):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(*quality_range))
        return Image.open(buffer)

    def gaussian_blur(self, img, radius_range=(0.5, 1.5)):
        radius = random.uniform(*radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))

    def color_jitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(img)

    def add_gaussian_noise(self, img, mean=0, std=0.01):
        img_tensor = F.to_tensor(img)
        noise = torch.randn_like(img_tensor) * std
        noisy_img = img_tensor + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        return F.to_pil_image(noisy_img)

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean()