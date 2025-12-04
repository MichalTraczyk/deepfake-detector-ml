import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from deepfake_detector.utils.metrics import evaluate_train_accuracy
from deepfake_detector.utils.checkpoint import save_checkpoint, load_checkpoint
from deepfake_detector.common import BalancedBatchSampler

def train_one_epoch(model, loader, optimizer, criterion, device, input_key=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        if input_key is not None:
            outputs = model(inputs[input_key])
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total



def train_k_fold(loaders: dict, params: dict, checkpoint_path: str, model_factory, input_key: str = None):
    train_loader = loaders['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
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

        model = model_factory().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        fold_checkpoint_path = checkpoint_path.replace('.pt', f'_fold{fold + 1}.pt')

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, fold_train_loader, optimizer, criterion, device, input_key=input_key)
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

        print("Memory allocated:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        print("Memory reserved:", torch.cuda.memory_reserved() / 1024 ** 2, "MB")

    final_model = model_factory().to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

    final_model, _, _ = load_checkpoint(final_model, optimizer, fold_checkpoint_path, device)

    return final_model
