import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from deepfake_detector.modules.cnn.model_cnn import MultiInputModel
from deepfake_detector.common import FFTImageDataset, BalancedBatchSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from deepfake_detector.utils.augment import AdvancedAugment
from deepfake_detector.utils.checkpoint import save_checkpoint, load_checkpoint
from deepfake_detector.utils.metrics import evaluate_model_metrics, evaluate_train_accuracy
from deepfake_detector.utils.train_utils import train_one_epoch


def create_dataloaders(params: dict):
    res = params['image_resolution']
    batch_size = params['batch_size']

    # Transforms
    advanced_augment = AdvancedAugment(prob=0.5)
    transform_rgb = transforms.Compose([
        transforms.Resize((res, res)),
        advanced_augment,
        transforms.ToTensor(),
    ])
    data_dir = "data/02_processed/"
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

def train(loaders: dict, params: dict):
    train_loader = loaders['train']
    val_loader = loaders['val']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiInputModel().to(device)

    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Training
    checkpoint_path = params['checkpoint_path_cnn']
    num_epochs = params['num_epochs']
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_train_accuracy(model,val_loader,criterion,device)
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

    return model

def run_final_evaluation(model, loaders: dict, params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = params['checkpoint_path_cnn']
    test_loader = loaders['test']
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)

    ev = evaluate_model_metrics(model,test_loader,device,transformation=torch.sigmoid)
    print(ev)
    return ev