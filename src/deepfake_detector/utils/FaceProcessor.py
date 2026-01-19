import cv2
import face_recognition
import numpy as np
import face_alignment
import time
import torch

class ImageFaceProcessor:
    """
    Klasa do wycinania i sprawdzanie jakości twarzy z klatek wideo.
    Wykorzystuje bibliotekę face_recognition (dlib).
    """
    def __init__(self, resolution):
        """
        Args:
            resolution (int): Docelowa rodzielczość do przeskalowania zdjęć.
        """
        self.resolution = resolution

    def is_blurry(self, image, lap_threshold=3.0, ten_threshold=15):
        """
        Funkcja do oceniania czy obraz jest rozmyty.
        Zastosowano:
        1. Wariancję Laplacian - do wykrywania krawędzi.
        2. Tenegrand - Sprawdza jak wyraźne są kontury i krawędzie

        Args:
            image: Obraz twarzy.
            lap_threshold (float): Próg odcięcia dla wariancji Laplacian.
            ten_threshold (float): Próg odcięcia dla wariancji Tenengrad.

        Returns:
            bool: True dla rozmytego, False dla ostrego.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        tenengrad_score = np.mean(sobel_magnitude)
        return laplacian_var < lap_threshold or tenengrad_score < ten_threshold

    def get_face(self, image):
        """
        Potok danych:
        -Detekcja
        -Wycięcie
        -Skalowanie
        -Weryfikacja czy obraz jest rozmyty

        Args:
            image: Klatka wideo.

        Returns:
            image | None: Zdjęcie z wyciętą twarzy lub None,
                          jeśli klatka nie przeszła weryfikacji.
        """
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return None
        top, right, bottom, left = face_locations[0]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, self.resolution)
        if self.is_blurry(face):
            return None
        return face

# [REAL] Total Frames: 10844
#    - Target Split: 80% / 20%
#    - Actual Split: 80.02% / 19.98%
#    - Train Videos: 670 | Test Videos: 166
# [FAKE] Total Frames: 58388
#    - Target Split: 80% / 20%
#    - Actual Split: 80.01% / 19.99%
#    - Train Videos: 4065 | Test Videos: 1033
# ✅ Split complete. Balanced by Frame Count, Grouped by Video ID.