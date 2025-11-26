import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from torchvision import transforms
from deepfake_detector.common import FFTImageDataset


def test_fft_dataset_structure():
    """
    Sprawdza czy __getitem__ zwraca poprawny słownik i kształty.
    """
    # 1. Mockujemy ImageFolder (udajemy dataset obrazkowy)
    mock_image_folder = MagicMock()
    mock_image_folder.__len__.return_value = 10

    # Symulujemy, że ImageFolder zwraca (PIL Image, Label)
    # Zamiast PIL Image damy numpy array (FFTImageDataset robi np.array(image))
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_label = 1
    mock_image_folder.__getitem__.return_value = (dummy_image, dummy_label)

    # 2. Transformacje
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # 3. Inicjalizacja
    dataset = FFTImageDataset(mock_image_folder, transform_rgb=transform)

    # 4. Test pobrania elementu
    item, label = dataset[0]

    # Sprawdzenia
    assert label == 1
    assert isinstance(item, dict)
    assert 'rgb_input' in item
    assert 'fft_input' in item

    # Sprawdzenie kształtów
    # RGB powinno być [3, 256, 256]
    assert item['rgb_input'].shape == (3, 256, 256)
    # FFT powinno być [1, 256, 256] (obliczane wewnątrz klasy)
    assert item['fft_input'].shape == (1, 256, 256)