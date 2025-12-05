# utils/metrics.py
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred, average="weighted"):
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix
    }


def evaluate_model_metrics(model, dataloader, device="cpu", threshold=0.5, transformation=None, input_key=None):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            if input_key is not None:
                outputs = model(inputs[input_key])
            else:
                outputs = model(inputs)

            preds = None

            if transformation is None:
                preds = (outputs > threshold).long()
            else:
                preds = (transformation(outputs) > threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_classification_metrics(all_labels, all_preds)


def evaluate_train_accuracy(model, loader, criterion, device="cpu",transformation=None, input_key: str = None):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.float().unsqueeze(1).to(device)

            if input_key is not None:
                outputs = model(inputs[input_key])
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = None
            if transformation is None:
                preds = (outputs > 0.5).long()
            else:
                preds = (transformation(outputs ) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
