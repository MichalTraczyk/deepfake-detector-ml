import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch Embedding: konwertuje obraz na sekwencję wektorów (embeddings).

    Klasa ta dzieli obraz na mniejsze kwadraty (patches), spłaszcza je
    na wektory o wymiarze `embed_dim` oraz dodaje token klasyfikacyjnt (CLS)
    i informacje o pozycji (position_embedings)

    Args:
        emebed_dim (int): Wymiar wektora reprezentacji.
        patch_size (int): Rozmiar kwadratu (patch) w tym wypadku 16.
        num_patches (int): Liczba patchy na które dzielimy obraz.
        dropout (float): Prawdopodobieństwo zerowania neuronów (dropout rate).
        in_channels (int): Liczba wejść (w tym wypadku 3 dla RGB).
    """
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)
        )

        # Token CLS: wektor uczący się klasy
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Osadzenia pozycyjne: informacja o lokalizacji łaty na obrazie
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Przetwarza obraz na sekwencję wektorów z informacją o pozycji (position_embedings).

        Args:
            x (torch.Tensor): Tensor wejściowy o kszatałcie (Batch, Channel, Height, Width).

        Returns:
            torch.Tensor: Tensor o kształcie (Batch, Num_patches, Emnbed_Dim).
        """
        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)

        x = self.patcher(x).permute(0, 2, 1)

        x = torch.cat([cls_token, x], dim=1)

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class ModelViT(nn.Module):
    """
    Architektura modelu Vision Transformer (ViT).

    Składa się z zmiany obrazu na sekwencję, warstw transformera czyli Samoatencji
    oraz głowicy klasyfikującej wyjście tokenu CLS.

    Args:
        img_size (int): Rozdzielczość obraz wejściowego.
        patch_size (int): Rozmiar łaty (patch).
        num_classes (int): Liczba klas wejściowych (w tym wypadku 2 dla Fake i Real).
        emebed_dim (int): Określa rozmiar danych wejściowych i wyjściowych każdego bloku enkodera.
        num_encoders (int): Głębokość sieci.
        num_heads (int): Liczba osobnych podprecesów Multi-Head Attention.
        hidden_dim (int): Wymiar warstwy MLP w bloku Enkodera.
        dropout (float): Poziom dropoutu.
        activation (str): Funkcja aktywacji w enkoderze (np. gelu czy relu).
        in_channels (int, optional): Liczba wejść.
    """
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_encoders, num_heads, hidden_dim, dropout,
                 activation, in_channels=3):
        super().__init__()

        # Sprawdzenie czy obraz jest podzielny przez patch_size
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2

        # Konwersja na sekwencje wektorów
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        # Blok Transformera
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # Głowica klasyfikacyjna
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        """
        Przepływ danych przez sieć

        Args:
            x (torch.Tensor): Obraz wejściowy (Batch, Channel, Height, Width).

        Returns: Logity klasyfikacyjne (Batch, Num_classes).
        """
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)

        x = self.mlp_head(x[:, 0, :])
        return x