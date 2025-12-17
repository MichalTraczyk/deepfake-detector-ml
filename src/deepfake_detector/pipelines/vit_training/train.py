import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from deepfake_detector.utils.metrics import evaluate_model_metrics
from deepfake_detector.modules.vit.model_vit import ModelViT
from deepfake_detector.modules.vit.model_utils import load_pretrained_weights
from deepfake_detector.utils.augment import AdvancedAugment
from deepfake_detector.common import ImageDataset, BalancedBatchSampler
from deepfake_detector.utils.train_utils import train_k_fold

def create_dataloaders(params: dict):
    res = params['image_resolution']
    batch_size = params['batch_size']
    data_dir = params['data_dir']

    transform_train = transforms.Compose([
        transforms.Resize((res, res)),
        AdvancedAugment(prob=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_folder = ImageFolder(os.path.join(data_dir, 'train'))
    test_folder = ImageFolder(os.path.join(data_dir, 'test'))

    train_dataset = ImageDataset(train_folder, transform_train)
    test_dataset = ImageDataset(test_folder, transform_val)

    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {"train": train_loader, "test": test_loader}


def run_training_loop(loaders: dict, params: dict, vit_params: dict):

    def create_pretrained_vit():
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

        model = load_pretrained_weights(model)
        return model

    final_model = train_k_fold(
        loaders=loaders,
        params=params,
        checkpoint_path="checkpoints/vit_pretrained.pt",
        model_factory=create_pretrained_vit,
        input_key="rgb_input",
        final_path="data/03_models/vit_model.pt"
    )

    return final_model


def run_final_evaluation(model, loaders: dict, params: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = loaders['test']
    ev = evaluate_model_metrics(model, test_loader, device, transformation=torch.sigmoid, input_key="rgb_input")
    print(ev)
    return ev