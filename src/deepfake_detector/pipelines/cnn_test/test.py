import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from deepfake_detector.modules.cnn.model_cnn import MultiInputModel
from deepfake_detector.common import FFTImageDataset
from deepfake_detector.utils.checkpoint import load_checkpoint


# ----------------------
# Branch wrappers
# ----------------------
import torch.nn as nn

class RGBBranchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.rgb_base = model.rgb_base

    def forward(self, x):
        return self.rgb_base(x)


class FFTBranchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fft_branch = model.fft_branch

    def forward(self, x):
        x = self.fft_branch(x).view(x.size(0), -1)
        return x


# ----------------------
# Grad-CAM runner
# ----------------------
def gradcam_on_branch(branch_model, input_tensor, target_layer, device):
    branch_model.eval().to(device)
    cam = GradCAM(model=branch_model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor.to(device),
                        targets=[ClassifierOutputTarget(0)])[0]

    base_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy() \
        if input_tensor.shape[1] == 3 else \
        np.repeat(input_tensor.squeeze().cpu().numpy()[..., None], 3, axis=2)

    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    vis = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)

    return vis


# ----------------------
# Load config and data
# ----------------------
def create_cnn_gradcam_visualization(params: dict):

    res = params['image_resolution']
    data_dir = "data/02_processed/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_rgb = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
    ])

    test_data = ImageFolder(os.path.join(data_dir, 'test'))
    test_dataset = FFTImageDataset(test_data, transform_rgb)

    # ----------------------
    # Load model checkpoint
    # ----------------------
    model = MultiInputModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = "saved/cnn_checkpoint.pt"
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    # ----------------------
    # Take one sample from test set
    # ----------------------
    (inputs, label) = test_dataset[6]
    rgb_tensor = inputs["rgb_input"].unsqueeze(0)
    fft_tensor = inputs["fft_input"].unsqueeze(0)

    # ----------------------
    # Run Grad-CAM for RGB branch
    # ----------------------
    rgb_wrapper = RGBBranchWrapper(model)
    target_layer_rgb = rgb_wrapper.rgb_base.features[-1][0]
    vis_rgb = gradcam_on_branch(rgb_wrapper, rgb_tensor, target_layer_rgb, device)

    # ----------------------
    # Run Grad-CAM for FFT branch
    # ----------------------
    fft_wrapper = FFTBranchWrapper(model)
    target_layer_fft = fft_wrapper.fft_branch[3]  # Conv2d(16 -> 32)
    vis_fft = gradcam_on_branch(fft_wrapper, fft_tensor, target_layer_fft, device)

    # ----------------------
    # Display side-by-side
    # ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(vis_rgb)
    axes[0].set_title("RGB Branch Grad-CAM")
    axes[0].axis("off")

    axes[1].imshow(vis_fft)
    axes[1].set_title("FFT Branch Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()

    return fig