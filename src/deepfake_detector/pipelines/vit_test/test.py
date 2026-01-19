import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_grad_cam.utils.image import show_cam_on_image
from deepfake_detector.common import ImageDataset
from deepfake_detector.modules.vit.model_vit import ModelViT
from pytorch_grad_cam import EigenCAM
from deepfake_detector.utils.metrics import evaluate_model_metrics, get_roc_plot



def load_vit_model_node(vit_params: dict, paths: dict, settings: dict) -> torch.nn.Module:
    """
    Ładowanie przetrenowego modelu ViT z punktu kontrolnego.

    Args:
        vit_params (dict): Parametry architektury.
        paths (dict): Słownik ścieżek (np. punktu kontrolnego).
        settings (dict): Słownik z ustawieniami uczenia i danych wejściowych.

    Returns:
        torch.nn.Module: Model w trybie ewaluacji.
    """
    checkpoint_path = paths["vit_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelViT(
        img_size=settings['image_resolution'],
        patch_size=vit_params['patch_size'],
        num_classes=1,
        embed_dim=vit_params['embed_dim'],
        num_encoders=vit_params['depth'],
        num_heads=vit_params['num_heads'],
        hidden_dim=vit_params['mlp_dim'],
        dropout=vit_params.get('dropout', 0.1),
        activation="gelu",
        in_channels=3
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def create_test_dataloader_node(settings: dict, params_preprocess:dict) -> DataLoader:
    """
    Przygotowanie loadera dla danych testowych.

    Returns:
        tuple: (loader_caleb, loader_ff)
            - osobnego loadery dla dwóch zbiorów
    """
    res = settings['image_resolution']
    batch_size = 8
    data_dir_celeb = params_preprocess["celeb_df_output"]
    data_dir_ff = params_preprocess["forensics_output"]

    transform_rgb = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = ImageFolder(os.path.join(data_dir_celeb, 'test'))
    test_dataset = ImageDataset(test_data, transform_rgb)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_dataff = ImageFolder(data_dir_ff)
    test_datasetff = ImageDataset(test_dataff, transform_rgb)
    test_loaderff = DataLoader(test_datasetff, batch_size=batch_size)

    test_loader.dataset_name = "Celeb"
    test_loaderff.dataset_name = "Face Forentics"
    return test_loader, test_loaderff


def run_evaluation(model, test_loader):
    """
    Ewaulacja na zbiorze testowym i generowanie raportu.

    Zapisywanie wykresu ROC oraz macierzy pomyłek.

    Args:
        model (torch.nn.Module): Model ViT.
        test_loader (DataLoader): Dane testowe.

    Returns:
        dict: Słownik z metrykami (AUC, F1, Accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid,input_key="rgb_input")
    ev["confusion_matrix"] = str(ev["confusion_matrix"])
    plot = get_roc_plot(roc_curve_fpr=ev["roc_curve_fpr"], roc_curve_tpr=ev["roc_curve_tpr"])
    plot.savefig(os.path.join("data/04_reporting/", test_loader.dataset_name))
    return ev

def vit_reshape_transform(tensor):
    """
    Odwracanie spłaszczenia spowrotem do postaci 2D i usunięcie tokenu klasyfikacyjnego.

    Args:
        tensor (torch.Tensor): Wyjście z warstwy atencji.

    Returns:
        torch.Tensor: Przekształcony tensor.
    """
    patch_embeddings = tensor[:, 1:, :]
    num_patches = patch_embeddings.shape[1]
    grid_size = int(np.sqrt(num_patches))
    result = patch_embeddings.transpose(1, 2)
    result = result.reshape(tensor.size(0), result.size(1), grid_size, grid_size)
    return result


def create_vit_gradcam_plot_node(model, loader):
    """
    Generacja mapy atencji przy użyciu EigenCAM.

    Args:
        model (nn.Module): Model ViT.
        loader (DataLoader): Dane do wizualizacji.

    Returns:
        fig (figure.Figure): Wykres z obrazami z nałożonymi heatmapami.
    """
    dataset = loader.dataset

    fake_indices = []
    real_indices = []

    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
        fake_indices = np.where(targets == 0)[0].tolist()
        real_indices = np.where(targets == 1)[0].tolist()
    else:
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label == 0:
                fake_indices.append(i)
            else:
                real_indices.append(i)

    num_per_class = 4
    selected_fake = random.sample(fake_indices, min(len(fake_indices), num_per_class))
    selected_real = random.sample(real_indices, min(len(real_indices), num_per_class))

    selected_indexes = selected_fake + selected_real
    rows, cols = 2, 4
    figsize = (16, 9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_layers = [model.encoder_blocks.layers[-1].norm1]

    cam = EigenCAM(model=model,
                   target_layers=target_layers,
                   reshape_transform=vit_reshape_transform)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for axis_idx, i in enumerate(selected_indexes):
        ax = axes[axis_idx]

        (item, label) = dataset[i]
        inputs = item["rgb_input"]
        rgb_tensor = inputs.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=rgb_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        img_tensor = inputs.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = std * img_tensor + mean
        img_rgb = np.clip(img_rgb, 0, 1)

        vis_rgb = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        ax.imshow(vis_rgb)

        l = "Fake" if label == 0 else "Real"

        ax.set_title(l)
        ax.axis("off")

    plt.tight_layout()

    save_path = "data/04_reporting/vit_eigencam_visualization.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    return fig