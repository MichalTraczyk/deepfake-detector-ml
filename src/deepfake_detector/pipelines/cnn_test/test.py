import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from deepfake_detector.modules.cnn.model_cnn import CnnModel
from deepfake_detector.common import ImageDataset
from deepfake_detector.test import gradcam_on_branch, count_parameters
from deepfake_detector.utils.checkpoint import load_checkpoint


# ----------------------
# Branch wrappers
# ----------------------
import torch.nn as nn

from deepfake_detector.utils.metrics import evaluate_model_metrics, get_roc_plot


class RGBBranchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.rgb_base = model.rgb_base

    def forward(self, x):
        return self.rgb_base(x)

def get_test_dataloader(params: dict):
    res = params['image_resolution']
    batch_size = params['batch_size']
    data_dir = "data/02_processed/"

    transform_rgb = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = ImageFolder(os.path.join(data_dir, 'test'))
    test_dataset = ImageDataset(test_data, transform_rgb)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def get_test_model(paths:dict):
    checkpoint_path = paths["cnn_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnModel()
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    model.to(device)
    model.eval()
    params, trainable = count_parameters(model)
    print("Liczba parametrow: " + params)
    return model

def run_evaluation(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid)
    ev["confusion_matrix"] = str(ev["confusion_matrix"])
    plot = get_roc_plot(roc_curve_fpr=ev["roc_curve_fpr"],roc_curve_tpr=ev["roc_curve_tpr"])
    plot.savefig("data/04_reporting/cnn_roc_plot.png")
    return ev


def create_cnn_gradcam_visualization(loader : DataLoader, model):
    #selected_indexes = [9847,10341,8907,13796,12202,13200]
    selected_indexes = [12200,13200,13796,13500,13400,12702]
    rows, cols = 2, 3
    figsize = (12, 12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rgb_wrapper = RGBBranchWrapper(model)
    target_layer_rgb = rgb_wrapper.rgb_base.features[-2][0]

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    axis = 0
    for i in selected_indexes:
        (inputs, label) = loader.dataset[i]
        rgb_tensor = inputs["rgb_input"].unsqueeze(0)
        vis_rgb = gradcam_on_branch(rgb_wrapper, rgb_tensor, target_layer_rgb, device)

        ax = axes[axis]
        axis += 1
        ax.imshow(vis_rgb)
        l = "Fake" if label == 0 else "Real"
        ax.set_title(f"{l}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/04_reporting/cnn_gradcam.png")
    return fig