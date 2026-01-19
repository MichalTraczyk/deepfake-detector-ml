# utils/checkpoint.py
import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    """
    Zapis punktu kontrolnego uczenia modelu.

    Args:
        model (nn.Module): Model CNN lub ViT.
        optimizer (torch.optim.Optimizer): Optymalizator z zapisanymi statystykami uczenia.
        epoch (int): Epoka na której zostało zatrzymane uczenie.
        path (str): Ścieżka zapisu modelu.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Wczytywanie stanu treningu z punktu kontrolnego, jeśli taki istnieje.

    Args:
        model (nn.Module): Model CNN lub ViT.
        optimizer (torch.optim.Optimizer): Zaincjowany optymalizator.
        path (str): Ścieżka do punktu kontrolnego.
        device (str, optional): Urządzenie docelowe "cpu" lub "cuda".

    Returns:
        tuple: (model, optimizer, start_epoch)
            - model (nn.Module): Model z wczytanymi z punktu kontrolnego wagami.
            - optimizer (torch.optim.Optimizer): Optymalizator z statystykami z punktu kontrolnego.
            - start_epoch (int): Epoka od której ma być wznowiony trening.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return model, optimizer, start_epoch
    else:
        print(f"⚠️ No checkpoint found at {path}. Starting fresh.")
        return model, optimizer, 0
