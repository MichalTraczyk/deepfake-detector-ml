import os
import configparser
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.cnn_fft.model_cnn_fft import MultiInputFFTModel
from src.models.common import FFTImageDataset, gradcam_on_branch
from src.utils.checkpoint import load_checkpoint


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
# Load config and data
# ----------------------
config_override = configparser.ConfigParser()
config_override.read('config.ini')

res = int(config_override["LearningSettings"]["ImageResolution"])
data_dir = "data_processed/"
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
model = MultiInputFFTModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
checkpoint_path = "saved/checkpoint_best.pt"
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

raw_img = inputs["rgb_input"].permute(1, 2, 0).cpu().numpy()
raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)

axes[0].imshow(raw_img)
axes[0].set_title("Raw image")
axes[0].axis("off")


axes[1].imshow(vis_rgb)
axes[1].set_title("Grad-CAM")
axes[1].axis("off")

plt.tight_layout()
plt.show()


from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        rgb_inputs = inputs["rgb_input"].to(device)
        fft_inputs = inputs["fft_input"].to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model({
            "rgb_input": rgb_inputs,
            "fft_input": fft_inputs
        })

        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)


accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
