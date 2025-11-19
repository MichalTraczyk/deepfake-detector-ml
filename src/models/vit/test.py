import os
import configparser
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model_vit import ModelViT
from src.models.common import FFTImageDataset
from src.utils.checkpoint import load_checkpoint


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

    vis = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)

    return vis, base_img



config_override = configparser.ConfigParser()
config_override.read('config.ini')

ls = config_override["LearningSettings"]
res = int(ls["ImageResolution"])
vs = config_override["ViT"]
patch_size = int(vs["patch_size"])
embed_dim = int(vs["embed_dim"])
depth = int(vs["depth"])
num_heads = int(vs["num_heads"])
mlp_dim = int(vs["mlp_dim"])

data_dir = "data_processed/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((res, res)),
    transforms.ToTensor()
])

test_data = ImageFolder(os.path.join(data_dir, 'test'))
test_dataset = FFTImageDataset(test_data, transform_test)

# ----------------------
# Load model checkpoint
# ----------------------
model = ModelViT(
    image_size=res, patch_size=patch_size, in_channels=3,
    embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim,
    dropout=0.0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
checkpoint_path = "saved/vit_checkpoint.pt"

print(f"Wczytywanie modelu z: {checkpoint_path}")
model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

# ----------------------
# Pick Sample
# ----------------------
SAMPLE_IDX = 12
(inputs, label) = test_dataset[SAMPLE_IDX]
rgb_tensor = inputs["rgb_input"].unsqueeze(0)

print(f"Analiza próbki: {SAMPLE_IDX}, Label: {label}")

# ----------------------
# Run Grad-CAM
# ----------------------

wrapper = ViTWrapper(model)

target_layer = [model.transformer_encoder.layers[-1].norm1]

grid_size = res // patch_size
reshape_lambda = lambda x: reshape_transform_vit(x, height=grid_size, width=grid_size)

vis_cam, base_img = run_gradcam(wrapper, rgb_tensor, target_layer, reshape_lambda, device)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(base_img)
axes[0].set_title(f"Original Image (Label: {label})")
axes[0].axis("off")

axes[1].imshow(vis_cam)
axes[1].set_title("ViT Attention (Grad-CAM)")
axes[1].axis("off")

plt.tight_layout()
plt.show()