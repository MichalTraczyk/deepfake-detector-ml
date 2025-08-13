import configparser
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from src.models.cnn.model_cnn import MultiInputModel
from src.models.common import FFTImageDataset, BalancedBatchSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.augment import AdvancedAugment
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.metrics import evaluate_model_metrics, evaluate_train_accuracy
from src.utils.train_utils import train_one_epoch

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
num_epochs = (int)(config_override["LearningSettings"]["CnnTrainingEpochs"])

model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

def train():
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(val_loader)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def evaluate(loader):
    return evaluate_train_accuracy(model,loader,criterion,device)

train()
ev = evaluate_model_metrics(model,test_loader,'cuda',transformation=torch.sigmoid)
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