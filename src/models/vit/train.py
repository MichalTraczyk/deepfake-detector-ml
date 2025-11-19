import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import configparser
from model_vit import ModelViT
from src.utils.model_utils import load_pretrained_weights
from src.utils.train_utils import train_one_epoch
from src.utils.metrics import evaluate_model_metrics, evaluate_train_accuracy
from src.models.common import FFTImageDataset, BalancedBatchSampler
from src.utils.augment import AdvancedAugment
from src.utils.checkpoint import save_checkpoint, load_checkpoint

config_override = configparser.ConfigParser()
config_override.read('config.ini')


ls = config_override["LearningSettings"]
res = int(ls["ImageResolution"])
batch_size = int(ls["TrainingBatchSize"])
num_epochs = int(ls["CnnTrainingEpochs"])

vs = config_override["ViT"]
vit_patch_size = int(vs["patch_size"])
vit_embed_dim = int(vs["embed_dim"])
vit_depth = int(vs["depth"])
vit_num_heads = int(vs["num_heads"])
vit_mlp_dim = int(vs["mlp_dim"])
vit_dropout = float(vs["dropout"])


learning_rate = 1e-4
checkpoint_path = "saved/vit_checkpoint.pt"
data_dir = "data_processed/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

advanced_augment = AdvancedAugment(prob=0.5)
transform_rgb = transforms.Compose([
    transforms.Resize((res, res)),
    advanced_augment,
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((res, res)),
    transforms.ToTensor()
])

train_data = ImageFolder(os.path.join(data_dir, 'train'))
val_data = ImageFolder(os.path.join(data_dir, 'val'))
test_data = ImageFolder(os.path.join(data_dir, 'test'))

train_dataset = FFTImageDataset(train_data, transform_rgb)
val_dataset = FFTImageDataset(val_data, transform_val)
test_dataset = FFTImageDataset(test_data, transform_val)

train_sampler = BalancedBatchSampler(train_dataset.dataset, batch_size)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model = ModelViT(
    image_size=res,
    patch_size=vit_patch_size,
    in_channels=3,
    embed_dim=vit_embed_dim,
    depth=vit_depth,
    num_heads=vit_num_heads,
    mlp_dim=vit_mlp_dim,
    dropout=vit_dropout
).to(device)

pos_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

if start_epoch == 0:
    print("Brak checkpointu treningowego.")
    model = load_pretrained_weights(model)
else:
    print(f"Wznowiono trening z epoki: {start_epoch}")


def evaluate(loader):
    return evaluate_train_accuracy(model, loader, criterion, device, transformation=torch.sigmoid)


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
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            print("Saved best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()

    print("\nLoading best model for final evaluation...")
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)

    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid)

    print("\n--- FINAL TEST METRICS ---")
    print(ev)