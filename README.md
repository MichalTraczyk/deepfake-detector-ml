# Praca Inżynierska

**Dokumentacja:** [Link do dokumentu](https://docs.google.com/document/d/1vwCZgo1lLyNIXctKjpwBRIgVVJ1DtH0ubrmZTCOeKFY/edit?usp=sharing)

## Zbiory danych (Datasets)

* **Dataset główny:** [Google Drive](https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj)
* **Dataset dodatkowy:** [Kaggle - FF-C23](https://www.kaggle.com/datasets/xdxd003/ff-c23)

### Przygotowanie danych
Dane należy wypakować do poniższej struktury katalogów:
```text
├── data/
│   ├── 01_raw/# Dane źródłowe
│   │   ├── Caleb-DF-v2/
│   │   └── face-forentics/
│   │       ├── FaceShifter/
│   │       └── original/
```
### Ustawienie środowiska

1. Instalacja bibliotek do trenowania na GPU
Pobierz i zainstaluj:

[CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

[cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10)

2. Instalacja zależności Python
Uruchom w terminalu:

```Bash

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
## Obsługa Kedro
Aby uruchomić wybrany potok: 
```bash
kedro run --pipeline NAZWA_POTOKU
```
### Dostępne potoki:
* preprocess – przygotowanie danych (wyodrębnianie pojedynczych klatek z filmów).
* training_cnn – trenowanie modelu CNN.
* training_vit – trenowanie modelu ViT (Vision Transformer).
* test_cnn – testowanie modelu CNN.
* test_vit – testowanie modelu ViT.

QUICKSTART

```bash
kedro run --pipeline preprocess
kedro run --pipeline training_cnn
kedro run --pipeline test_cnn
```
Finalna struktura plików projektu

```text
.
├── checkpoints/# Checkpointy do wznawiania trenowania modeli
├── data/
│   ├── 01_raw/# Dane źródłowe
│   │   ├── Caleb-DF-v2/
│   │   └── face-forentics/
│   │       ├── FaceShifter/
│   │       └── original/
│   ├── 02_processed/# Dane przetworzone
│   │   ├── celeb_df/
│   │   │   ├── test/
│   │   │   └── train/
│   │   └── forencics/
│   │       ├── fake/
│   │       └── real/
│   ├── 03_models/# Wyeksportowane modele (.pt)
│   │   ├── cnn_model.pt
│   │   └── vit_model.pt
│   └── 04_reporting/ # Wyniki: wykresy, metryki
...