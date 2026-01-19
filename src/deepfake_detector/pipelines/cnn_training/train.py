import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from deepfake_detector.modules.cnn.model_cnn import CnnModel
from deepfake_detector.common import BalancedBatchSampler, ImageDataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from deepfake_detector.utils.augment import AdvancedAugment
from deepfake_detector.utils.checkpoint import save_checkpoint, load_checkpoint
from deepfake_detector.utils.metrics import evaluate_model_metrics, evaluate_train_accuracy
from deepfake_detector.utils.train_utils import train_k_fold


def create_dataloaders(params: dict):
    """
        Przygotowuje dane do treningu i testów.

        Tworzy DataLoaders z odpowiednimi transformacjami:
        - Trening i Test: augmentacja, skalowanie i normalizacja.
        Balansuje klasy w treningu za pomocą BalancedBatchSampler (50% Real i 50% Fake).

        Args:
            params (dict): Słownik zawierający parametry uczenia i danych wejściowych.

        Returns:
            dict: Słownik z loaderami dla kluczy 'train' i 'test'.
        """
    res = params['image_resolution']
    batch_size = params['batch_size']

    # Transforms
    advanced_augment = AdvancedAugment(prob=0.5)
    transform_rgb = transforms.Compose([
        transforms.Resize((res, res)),
        advanced_augment,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_dir = params['data_dir']
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
    """
    Tworzenie modelu CNN

    Returns:
        nn.Module: CNN model do uczenia.
    """
    model = CnnModel()

    return model


def train(loaders: dict, params: dict):
    """
    Uczenie modelu CNN.

    Walidacja krzyżowa (K-Fold) i trening na całości danych.

    Args:
        loaders (dict): Dane treningowe i testowe.
        params (dict): Parametry treningowe.

    Returns:
        torch.nn.Module: Wytrenowany model.
    """
    final_model = train_k_fold(
        loaders=loaders,
        params=params,
        checkpoint_path="checkpoints/cnn_pretrained.pt",
        model_factory=create_model,
        final_path="data/03_models/cnn_model.pt"
    )

    return final_model


def run_final_evaluation(model, loaders: dict, params: dict):
    """
    Funkcja do ewaluacji modelu CNN.

    Przeprowadzanie predykcji na danych testowych i obliczanie metryk skuteczności modelu.

    Args:
        model (nn.Module): Wytrenowany model CNN.
        loaders (dict): Dane treningowe i testowe.

    Returns:
        ev (dict): Metryki.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = loaders['test']
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid)
    print(ev)
    return ev
