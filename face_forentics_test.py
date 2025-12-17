import configparser
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from deepfake_detector.modules.cnn.model_cnn import CnnModel
from deepfake_detector.common import ImageDataset
from deepfake_detector.utils.checkpoint import load_checkpoint
from deepfake_detector.utils.metrics import evaluate_model_metrics, get_roc_plot

config_override = configparser.ConfigParser()
config_override.read('config.ini')

res = (int)(config_override["LearningSettings"]["ImageResolution"])
source_path = "data/face_forentics_processed/"

def get_test_dataloader(params: dict):
    res = params['ImageResolution']
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
    return model

def run_evaluation(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid)
    ev["confusion_matrix"] = str(ev["confusion_matrix"])
    plot = get_roc_plot(roc_curve_fpr=ev["roc_curve_fpr"],roc_curve_tpr=ev["roc_curve_tpr"])
    plot.savefig("data/04_reporting/face_forentics_roc_curve.png")
    return ev

