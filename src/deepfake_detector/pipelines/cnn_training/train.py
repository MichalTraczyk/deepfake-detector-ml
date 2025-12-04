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
from deepfake_detector.utils.train_utils import train_k_fold


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


def create_model():
    model = MultiInputModel()

    return model


def train(loaders: dict, params: dict):
    final_model = train_k_fold(
        loaders=loaders,
        params=params,
        checkpoint_path="checkpoints/cnn_pretrained.pt",
        model_factory=create_model
    )

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