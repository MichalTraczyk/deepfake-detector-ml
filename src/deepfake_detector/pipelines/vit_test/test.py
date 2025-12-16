import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from deepfake_detector.common import ImageDataset
from deepfake_detector.modules.vit.model_utils import load_pretrained_weights
from deepfake_detector.modules.vit.model_vit import ModelViT
from deepfake_detector.utils.checkpoint import load_checkpoint
from deepfake_detector.utils.metrics import evaluate_model_metrics, get_roc_plot


# ----------------------
# ViT Reshape Transform
# ----------------------
def reshape_transform_vit(tensor, height=16, width=16):
    result = tensor[:, 1:, :]

    result = result.reshape(tensor.size(0), height, width, tensor.size(2))

    result = result.permute(0, 3, 1, 2)
    return result


# ----------------------
# Wrapper
# ----------------------

class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({'rgb_input': x})


# ----------------------
# Grad-CAM runner
# ----------------------
def run_gradcam(model_wrapper, input_tensor, target_layers, reshape_function, device):
    model_wrapper.eval().to(device)

    cam = GradCAM(
        model=model_wrapper,
        target_layers=target_layers,
        reshape_transform=reshape_function
    )

    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)[0]

    base_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)

    vis = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)

    return vis, base_img


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

def get_test_model(params:dict,vit_params: dict, paths:dict):
    checkpoint_path = paths["vit_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelViT(
        img_size=params['image_resolution'],
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
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    model.to(device)
    model.eval()
    return model
def run_evaluation(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid, input_key="rgb_input")
    ev["confusion_matrix"] = str(ev["confusion_matrix"])
    plot = get_roc_plot(roc_curve_fpr=ev["roc_curve_fpr"],roc_curve_tpr=ev["roc_curve_tpr"])
    plot.savefig("data/04_reporting/vit_roc_plot.png")
    return ev

def create_vit_gradcam_visualization(model, test_loader : DataLoader, learning_params: dict, vit_params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = learning_params['image_resolution']
    patch_size = vit_params['patch_size']
    grid_size = res // patch_size
    reshape_lambda = lambda x: reshape_transform_vit(x, height=grid_size, width=grid_size)

    rows, cols = 2, 3
    figsize = (12, 12)
    selected_indexes = [9847,10341,8907,13796,12202,13200]

    wrapper = ViTWrapper(model)
    model.to(device)
    target_layer = [model.encoder_blocks.layers[-1].norm1]

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    axis = 0
    for i in selected_indexes:
        (inputs, label) = test_loader.dataset[i]
        rgb_tensor = inputs["rgb_input"].unsqueeze(0)
        vis_cam, base_img = run_gradcam(
            model_wrapper=wrapper,
            input_tensor=rgb_tensor,
            target_layers=target_layer,
            reshape_function=reshape_lambda,
            device=device
        )

        ax = axes[axis]
        axis += 1
        ax.imshow(vis_cam)
        ax.set_title(f"Sample {i} (True Label: {label})")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/04_reporting/vit_attention.png")
    return fig