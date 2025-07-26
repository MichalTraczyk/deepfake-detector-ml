import configparser
import os
import cv2
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import random_split
from torchvision.transforms.functional import to_tensor

import os
print(os.getcwd())


config_override = configparser.ConfigParser()
config_override.read('config.ini')

res = (int)(config_override["LearningSettings"]["ImageResolution"])
batch_size = (int)(config_override["LearningSettings"]["TrainingBatchSize"])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Transforms
transform_rgb = transforms.Compose([
    transforms.Resize((res, res)),
    transforms.ToTensor(),
])

data_dir = "data_processed/"
train_data = ImageFolder(os.path.join(data_dir, 'train'))
val_data = ImageFolder(os.path.join(data_dir, 'val'))
test_data = ImageFolder(os.path.join(data_dir, 'test'))

train_dataset = FFTImageDataset(train_data, transform_rgb)
val_dataset = FFTImageDataset(val_data, transform_rgb)
test_dataset = FFTImageDataset(test_data, transform_rgb)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model definition
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

model = MultiInputModel().to(device)

# Loss and optimizer
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

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
checkpoint_path = "saved/checkpoint.pt"
resume_epochs = int(input("Resume? How many more epochs, 0 if fresh train: "))
start_epoch = 0
num_epochs = 30 if resume_epochs == 0 else resume_epochs

if resume_epochs > 0 and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

# Class weights
def compute_class_weights(loader):
    labels = []
    for _, label in loader:
        labels.extend(label.numpy())
    counter = Counter(labels)
    total = sum(counter.values())
    return {
        0: total / (2 * counter[0]),
        1: total / (2 * counter[1])
    }

class_weights_dict = compute_class_weights(train_loader)
class_weights = torch.tensor([class_weights_dict[0], class_weights_dict[1]], device=device)

def train():
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            inputs, labels = batch
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(val_loader)
        train_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

train()
test_loss, test_acc = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.2f}")
