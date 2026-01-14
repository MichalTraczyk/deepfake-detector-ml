import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
