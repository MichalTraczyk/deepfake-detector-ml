import torch.nn as nn
from torchvision import models, transforms
import torch
class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()

        # FFT branch
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # RGB branch using pretrained Xception-like model
        self.rgb_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.rgb_base.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        rgb_input = inputs['rgb_input']
        fft_input = inputs['fft_input']

        x_rgb = self.rgb_base(rgb_input)
        x_fft = self.fft_branch(fft_input).view(rgb_input.size(0), -1)

        combined = torch.cat([x_rgb, x_fft], dim=1)
        return self.fc(combined)