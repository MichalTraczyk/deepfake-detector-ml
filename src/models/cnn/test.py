import os
import configparser
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.models.cnn.model_cnn import MultiInputModel
from src.models.common import ImageDataset, gradcam_on_branch
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
test_dataset = ImageDataset(test_data, transform_rgb)


# ----------------------
# Load model checkpoint
# ----------------------
model = MultiInputModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
checkpoint_path = "saved/checkpoint_cnn.pt"
model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)


# ----------------------
# Take one sample from test set
# ----------------------
(inputs, label) = test_dataset[3]
rgb_tensor = inputs["rgb_input"].unsqueeze(0)


# ----------------------
# Run Grad-CAM for RGB branch
# ----------------------
rgb_wrapper = RGBBranchWrapper(model)
target_layer_rgb = rgb_wrapper.rgb_base.features[-2][0]
vis_rgb = gradcam_on_branch(rgb_wrapper, rgb_tensor, target_layer_rgb, device)

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

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        rgb_inputs = inputs["rgb_input"].to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model({"rgb_input": rgb_inputs})
        preds = torch.sigmoid(outputs)  # convert logits to probabilities
        predicted = (preds > 0.5).float()  # threshold at 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")