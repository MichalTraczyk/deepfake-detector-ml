# utils/checkpoint.py
import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return model, optimizer, start_epoch
    else:
        print(f"⚠️ No checkpoint found at {path}. Starting fresh.")
        return model, optimizer, 0
