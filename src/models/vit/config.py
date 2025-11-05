import configparser
from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int
    patch_size: int
    in_channels: int
    num_classes: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_dim: int
    dropout: float

@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int


def load_config_from_ini(filepath: str) -> (ViTConfig, TrainConfig):
    parser = configparser.ConfigParser()
    parser.read(filepath)

    vit_section = parser['ViT'] if 'ViT' in parser else {}

    vit_config = ViTConfig(
        image_size=vit_section.getint('image_size', 256),
        patch_size=vit_section.getint('patch_size', 16),
        in_channels=vit_section.getint('in_channels', 3),
        num_classes=vit_section.getint('num_classes', 10),
        embed_dim=vit_section.getint('embed_dim', 768),
        depth=vit_section.getint('depth', 12),
        num_heads=vit_section.getint('num_heads', 12),
        mlp_dim=vit_section.getint('mlp_dim', 3072),
        dropout=vit_section.getfloat('dropout', 0.1)
    )

    train_section = parser['Training'] if 'Training' in parser else {}

    train_config = TrainConfig(
        learning_rate=train_section.getfloat('learning_rate', 0.0001),
        batch_size=train_section.getint('batch_size', 32),
        num_epochs=train_section.getint('num_epochs', 20)
    )

    return vit_config, train_config