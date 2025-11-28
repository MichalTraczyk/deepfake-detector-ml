import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
from collections import Counter
from tqdm import tqdm
import numpy as np

from deepfake_detector.modules.vit.model_vit import ModelViT
from deepfake_detector.modules.vit.model_utils import load_pretrained_weights
from deepfake_detector.utils.checkpoint import save_checkpoint, load_checkpoint
from deepfake_detector.utils.augment import AdvancedAugment


class CleanImageDataset(Dataset):

    def __init__(self, image_folder: ImageFolder, transform_rgb=None):
        self.dataset = image_folder
        self.transform_rgb = transform_rgb

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_float = float(label)

        if self.transform_rgb:
            rgb_input = self.transform_rgb(image)
        else:
            rgb_input = transforms.ToTensor()(image)

        return {"rgb_input": rgb_input}, label_float


def make_weighted_sampler(image_folder: ImageFolder):
    targets = [s[1] for s in image_folder.samples]
    class_count = Counter(targets)
    print(f"   [Sampler] Liczebność klas: {class_count} (0=Fake, 1=Real)")

    class_weights = {cls: 1.0 / (count + 1e-6) for cls, count in class_count.items()}
    samples_weights = np.array([class_weights[t] for t in targets], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(samples_weights),
        num_samples=len(samples_weights),
        replacement=True
    )
    return sampler


def create_dataloaders(params: dict):
    print("--- Tworzenie DataLoaderów (Fake=0, Real=1) ---")
    res = params['image_resolution']
    batch_size = params['batch_size']
    data_dir = params['data_dir']

    transform_train = transforms.Compose([
        transforms.Resize((res, res)),
        AdvancedAugment(prob=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')

    train_folder = ImageFolder(train_path)
    val_folder = ImageFolder(val_path)
    test_folder = ImageFolder(test_path)

    train_dataset = CleanImageDataset(train_folder, transform_train)
    val_dataset = CleanImageDataset(val_folder, transform_val)
    test_dataset = CleanImageDataset(test_folder, transform_val)

    train_sampler = None
    shuffle = True

    if params.get('use_weighted_sampler', True):
        train_sampler = make_weighted_sampler(train_folder)
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)

    for inputs, labels in pbar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == labels).sum().item()

        all_preds.extend(preds.detach().cpu().numpy().ravel().tolist())
        all_labels.extend(labels.cpu().numpy().ravel().tolist())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)

    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy().ravel().tolist())
            all_labels.extend(labels.cpu().numpy().ravel().tolist())

    if total_samples == 0: return 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)

    return avg_loss, acc, f1

def run_training_loop(loaders: dict, params: dict, vit_params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kedro: Start treningu (Fake=0, Real=1) na: {device}")

    model = ModelViT(
        image_size=params['image_resolution'],
        patch_size=vit_params['patch_size'],
        in_channels=3,
        embed_dim=vit_params['embed_dim'],
        depth=vit_params['depth'],
        num_heads=vit_params['num_heads'],
        mlp_dim=vit_params['mlp_dim'],
        dropout=vit_params.get('dropout', 0.1)
    ).to(device)

    model = load_pretrained_weights(model)

    criterion = nn.BCEWithLogitsLoss()

    lr = params.get('learning_rate', 3e-5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    checkpoint_path = params['checkpoint_path_vit']
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    if start_epoch == 0:
        print("Start od zera (pretrained weights).")
    else:
        print(f"Wznowienie od epoki {start_epoch}.")

    best_val_loss = float('inf')
    patience = params.get('patience', 5)
    patience_counter = 0
    num_epochs = params['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        t_loss, t_acc, t_f1 = train_one_epoch(model, loaders['train'], optimizer, criterion, device)
        v_loss, v_acc, v_f1 = evaluate(model, loaders['val'], criterion, device)

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Train: loss={t_loss:.4f} acc={t_acc:.4f} f1_fake={t_f1:.4f} | "
              f"Val: loss={v_loss:.4f} acc={v_acc:.4f} f1_fake={v_f1:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            print("Zapisano checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

        scheduler.step()

    return model


def run_final_evaluation(model, loaders: dict, params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Finalna ewaluacja...")

    checkpoint_path = params['checkpoint_path_vit']
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc, test_f1 = evaluate(model, loaders['test'], criterion, device)

    results = {"loss": test_loss, "acc": test_acc, "f1_fake": test_f1}
    print(f"\n--- WYNIKI TESTU ---\n{results}")

    return results