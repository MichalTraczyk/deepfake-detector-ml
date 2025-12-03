import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from deepfake_detector.modules.cnn.model_cnn import MultiInputModel
from deepfake_detector.common import BalancedBatchSampler, ImageDataset
from torch.utils.data import DataLoader, Subset
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
    test_data = ImageFolder(os.path.join(data_dir, 'test'))

    train_dataset = ImageDataset(train_data, transform_rgb)
    test_dataset = ImageDataset(test_data, transform_rgb)

    train_sampler = BalancedBatchSampler(train_dataset.dataset, batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {
        "train": train_loader,
        "test": test_loader
    }

def train(loaders: dict, params: dict):
    train_loader = loaders['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    base_checkpoint_path = params['checkpoint_path_cnn']
    k_folds = params['k_folds']

    full_dataset = train_loader.dataset
    all_targets = np.array(full_dataset.dataset.targets)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_checkpoint_path = ""
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_targets)), all_targets)):
        print(f"\n{'=' * 15} FOLD {fold + 1}/{k_folds} {'=' * 15}")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_labels_fold = all_targets[train_idx]
        train_sampler = BalancedBatchSampler(train_subset, batch_size, custom_labels=train_labels_fold)

        fold_train_loader = DataLoader(train_subset, batch_sampler=train_sampler)
        fold_val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = MultiInputModel().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        fold_checkpoint_path = base_checkpoint_path.replace('.pt', f'_fold{fold + 1}.pt')

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, fold_train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_train_accuracy(model, fold_val_loader, criterion, device)

            print(f"Fold {fold + 1} [Ep {epoch + 1}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, fold_checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered for Fold {fold + 1}")
                    break

        del model, optimizer, fold_train_loader, fold_val_loader
        torch.cuda.empty_cache()

    final_model = MultiInputModel().to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)
    final_model, _, _ = load_checkpoint(final_model, optimizer, fold_checkpoint_path, device)

    return final_model

def run_final_evaluation(model, loaders: dict, params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = params['checkpoint_path_cnn']
    test_loader = loaders['test']
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)

    ev = evaluate_model_metrics(model,test_loader,device,transformation=torch.sigmoid)
    print(ev)
    return ev