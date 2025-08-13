from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def gradcam_rgb_branch(model, rgb_img_path, fft_tensor, device='cuda'):
    """
    Visualize Grad-CAM for the RGB branch.
    """
    model.eval().to(device)

    # Load and preprocess RGB image
    orig = cv2.imread(rgb_img_path)[:, :, ::-1]  # BGR → RGB
    rgb_resized = cv2.resize(orig, (224, 224))
    rgb_tensor = preprocess_image(
        rgb_resized.astype(np.float32) / 255.0,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    inputs = {
        'rgb_input': rgb_tensor.to(device),
        'fft_input': fft_tensor.to(device)
    }

    # Grad-CAM target layer for RGB branch
    target_layer = model.rgb_base.features[-1][0]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == 'cuda'))
    grayscale_cam = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(0)])[0]
    vis = show_cam_on_image(rgb_resized.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)

    plt.imshow(vis)
    plt.title("RGB Branch Grad-CAM")
    plt.axis("off")
    plt.show()


def gradcam_fft_branch(model, rgb_tensor, fft_input, device='cuda'):
    """
    Visualize Grad-CAM for the FFT branch.
    """
    model.eval().to(device)

    # Ensure FFT input is tensor [B, 1, H, W]
    if isinstance(fft_input, np.ndarray):
        fft_tensor = torch.from_numpy(fft_input).unsqueeze(0).unsqueeze(0)
    else:
        fft_tensor = fft_input

    inputs = {
        'rgb_input': rgb_tensor.to(device),
        'fft_input': fft_tensor.to(device)
    }

    # Grad-CAM target layer for FFT branch
    target_layer = model.fft_branch[3]  # Second Conv2d layer
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == 'cuda'))
    grayscale_cam = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(0)])[0]

    # FFT image to RGB for overlay
    fft_img_3ch = np.repeat(fft_tensor.squeeze().cpu().numpy()[:, :, None], 3, axis=2)
    vis = show_cam_on_image(fft_img_3ch / fft_img_3ch.max(), grayscale_cam, use_rgb=True)

    plt.imshow(vis)
    plt.title("FFT Branch Grad-CAM")
    plt.axis("off")
    plt.show()