import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gradcam_on_branch(branch_model, input_tensor, target_layer, device):
    """
    Generowanie mapy aktywacji Grad-Cam, czyli wizualacja na czym się skupia model.

    Args:
        branch_model (torch.nn.Module): Model lub jego gałąż (np. encoder).
        input_tensor (torch.Tensor): Przetworzony obraz wejściowy.
        target_layer (torch.nn.Module): Warstwa z której pobierany jest gradient.
        device (torch.device): Urządzenie obliczeniowe "cpu" lub "cuda".

    Returns:
        np.ndarray: Obraz RGB z mapą ciepła.
    """
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


def count_parameters(model):
    """
    Zliczanie parametrów modelu, do analizy złożoności obliczeniowej modelu
    oraz weryfikacji jakie warstwy są trenowalne.

    Args:
        model (torch.nn.Module): Analizowany model.

    Returns:
        tuple: (total_params, trainable_params):
            - total_params (int): Całkowita liczba wag i biasów.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
