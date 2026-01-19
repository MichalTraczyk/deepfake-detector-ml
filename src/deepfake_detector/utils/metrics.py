# utils/metrics.py
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, \
    roc_curve


def compute_classification_metrics(y_true, y_pred, average="weighted"):
    """
    Obliczanie metryk klasyfikacji przy użyciu biblioteki scikit-learn.
    
    Args:
        y_true (array): Rzeczywiste etykiety klas.
        y_pred (array): Przewidywane etykiety klas.
        average (str): Metoda do uśredniania klasyfikacji.

    Returns:
        dict: Metryki: presision, recall, f1_score, accuracy_score i confusion matrix.
    """
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


def calculate_roc_auc(y_true, y_score):
    """
    Obliczanie pola pod krzywą ROC (AUC).

    Args:
        y_true (array): Rzeczywiste etykiety klas.
        y_score (array): Logity zwrócone przed model.

    Returns:
        dict: Wartość AUC
    """

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        print("Warning: Only one class present in y_true. ROC AUC set to 0.5.")
        roc_auc = 0.5

    # Wyliczanie punktów krzywej ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "roc_auc": float(roc_auc),
        "roc_curve_fpr": fpr.tolist(),
        "roc_curve_tpr": tpr.tolist()
    }


def evaluate_model_metrics(model, dataloader, device="cpu", threshold=0.5, transformation=None, input_key=None):
    """
    Pętla do ewaluacji modelu.

    Args:
        model (nn.Module): Model do ewaluacji.
        dataloader (DataLoader): Dataloader danych testowych.
        device (str, optional): Urządzenie obliczeniowe "cpu" lub "cuda".
        threshold (float, optional): Próg odcięcia dla klasyfikacji binarnej.
        transformation (callable, optional): Funkcja wyjściowa np. sigmoid.
        input_key (str, optional): Klucz słownika.

    Returns:
        dict: Słownik z metrykami klasyfikacji, ROC oraz czasem ewaluacji.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_scores = []
    time_spent_evaluating = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            if input_key is not None:
                outputs = model(inputs[input_key])
            else:
                outputs = model(inputs)

            if transformation is not None:
                outputs = transformation(outputs)

            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            preds = (outputs > threshold).long()
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            time_spent_evaluating += end_time - start_time
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
    roc_metrics = calculate_roc_auc(all_labels, all_scores)
    eval_time = {"evaluation_time": time_spent_evaluating}
    classification_metrics = compute_classification_metrics(all_labels, all_preds)
    classification_metrics.update(eval_time)
    classification_metrics.update(roc_metrics)
    print("Evaluating: " + str(len(all_preds)) + " took " + str(time_spent_evaluating) + " seconds")
    return classification_metrics


def evaluate_train_accuracy(model, loader, criterion, device="cpu", transformation=None, input_key: str = None):
    """
    Uproszczona funkcja do ewaluacji modelu podczas treningu.

    Args:
        model (nn.Module): Model do ewaluacji.
        loader (DataLoader): Loader danych.
        criterion (loss function): Funkcja straty.
        device (str): Urządzenie obliczenie "cpu" lub "cuda".

    Returns:
        tuple: (avarage_loss, accuracy).
    """
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
                preds = (transformation(outputs) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_roc_plot(roc_curve_fpr, roc_curve_tpr):
    """
    Generowanie wykresu dla krzywej ROC.

    Args:
        roc_curve_tpr (list): Współczynnik Prawdziwie Pozytywnych.
        roc_curve_fpr (list): Współczynnik Fałszywie Pozytywnych.

    Returns:
        matplotlib.figure.Figure: Wykres z krzywą ROC.
    """
    roc_auc = np.trapz(roc_curve_tpr, roc_curve_fpr)
    fig, ax = plt.subplots(figsize=(8, 8))

    # 3. Plot the ROC curve
    ax.plot(
        roc_curve_fpr,
        roc_curve_tpr,
        color='darkorange',
        lw=2,
        label=f'Krzywa ROC'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='(AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Współczynnik Fałszywie Pozytywnych (FPR)')
    ax.set_ylabel('Współczynnik Prawdziwie Pozytywnych (TPR) / Czułość')
    ax.set_title('Krzywa ROC')
    ax.legend(loc="lower right")
    ax.grid(True)

    # 6. Return the figure object
    return fig


13840
