import torch.nn as nn
from torchvision import models, transforms
import torch
class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()
        self.rgb_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.rgb_base.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        rgb_input = inputs['rgb_input']
        x_rgb = self.rgb_base(rgb_input)
        return self.fc(x_rgb)