import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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


def create_vit_gradcam_visualization(model, loaders: dict, learning_params: dict, vit_params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generowanie Grad-CAM na urządzeniu: {device}")

    test_loader = loaders['test']
    dataset = test_loader.dataset

    SAMPLE_IDX = 12

    inputs, label = dataset[SAMPLE_IDX]
    rgb_tensor = inputs["rgb_input"].unsqueeze(0)

    print(f"Analiza próbki: {SAMPLE_IDX}, Label: {label}")

    res = learning_params['image_resolution']
    patch_size = vit_params['patch_size']

    grid_size = res // patch_size

    reshape_lambda = lambda x: reshape_transform_vit(x, height=grid_size, width=grid_size)

    wrapper = ViTWrapper(model)

    model.to(device)
    target_layer = [model.transformer_encoder.layers[-1].norm1]

    vis_cam, base_img = run_gradcam(
        model_wrapper=wrapper,
        input_tensor=rgb_tensor,
        target_layers=target_layer,
        reshape_function=reshape_lambda,
        device=device
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(base_img)
    axes[0].set_title(f"Original Image (Label: {label})")
    axes[0].axis("off")

    axes[1].imshow(vis_cam)
    axes[1].set_title("ViT Attention (Grad-CAM)")
    axes[1].axis("off")

    plt.tight_layout()

    return fig