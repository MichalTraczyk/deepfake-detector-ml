import configparser
import time
from collections import Counter
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from models.cnn.model import MultiInputModel
from models.common import FFTImageDataset, FocalLoss, AdvancedAugment, BalancedBatchSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from models.test import evaluate_model

config_override = configparser.ConfigParser()
config_override.read('config.ini')

res = (int)(config_override["LearningSettings"]["ImageResolution"])
batch_size = (int)(config_override["LearningSettings"]["TrainingBatchSize"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transforms
advanced_augment = AdvancedAugment(prob=0.5)
transform_rgb = transforms.Compose([
    transforms.Resize((res, res)),
    advanced_augment,
    transforms.ToTensor(),
])
data_dir = "data_processed/"
train_data = ImageFolder(os.path.join(data_dir, 'train'))
val_data = ImageFolder(os.path.join(data_dir, 'val'))
test_data = ImageFolder(os.path.join(data_dir, 'test'))

train_dataset = FFTImageDataset(train_data, transform_rgb)
val_dataset = FFTImageDataset(val_data, transform_rgb)
test_dataset = FFTImageDataset(test_data, transform_rgb)

train_sampler = BalancedBatchSampler(train_dataset.dataset, batch_size)


train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)


val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model = MultiInputModel().to(device)

pos_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Training
checkpoint_path = "saved/checkpoint.pt"
start_epoch = 0
num_epochs = (int)(config_override["LearningSettings"]["CnnTrainingEpochs"])

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

def train():
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False):
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
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = evaluate(val_loader)
        train_acc = correct / total
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss / total:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


train()
ev = evaluate_model(model,test_loader,'cuda')
print(ev)

# Get a sample from FFTImageDataset
# (inputs, label) = dataset[0]
# rgb_tensor = inputs["rgb_input"].unsqueeze(0).to(device)
# fft_tensor = inputs["fft_input"].unsqueeze(0).to(device)
#
# # Grad-CAM for RGB branch
# cam_rgb = gradcam_rgb_branch(model, rgb_tensor)
# rgb_img = rgb_tensor.squeeze().permute(1,2,0).cpu().numpy()
# rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize for visualization
# cam_on_rgb = show_cam_on_image(rgb_img, cam_rgb, use_rgb=True)
#
# # Grad-CAM for FFT branch
# cam_fft = gradcam_fft_branch(model, fft_tensor)
# fft_img = fft_tensor.squeeze().cpu().numpy()
# fft_img = (fft_img - fft_img.min()) / (fft_img.max() - fft_img.min())
# fft_img_rgb = np.repeat(fft_img[..., None], 3, axis=2)  # grayscale → RGB
# cam_on_fft = show_cam_on_image(fft_img_rgb, cam_fft, use_rgb=True)