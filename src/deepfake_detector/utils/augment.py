import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import io

class AdvancedAugment:
    """
    Klasa do augmentacji zdjęć przez 4 filtry.

    Symuluje wady wideo, takie jak kompresja, rozmycie, szum
    czy manipulacja jasnością, kontrastem i nasyceniem kolorów.
    """
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Prawodpodobieństwo aplikowania poszczególnych augmentacji.
        """
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = self.jpeg_compress(img)
        if random.random() < self.prob:
            img = self.gaussian_blur(img)
        if random.random() < self.prob:
            img = self.color_jitter(img)
        if random.random() < self.prob:
            img = self.add_gaussian_noise(img)
        return img

    def jpeg_compress(self, img, quality_range=(30, 70)):
        """
        Symulowanie kompresji JPEG, aby popsuć jakość obrazu.

        Args:
            img: Obraz wejściowy.
            quality_range (tuple): Zakres jakości kompresji.

        Returns:
            PIL.Image : Obraz po kompresji JPEG.
        """
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(*quality_range))
        return Image.open(buffer)

    def gaussian_blur(self, img, radius_range=(0.5, 1.5)):
        """
        Rozmywanie obrazu.

        Args:
            img: Obraz wejściowy.
            radius_range (tuple): Zakres rozmycia.

        Returns:
            PIL.Image : Obraz po rozmyciu.
        """
        radius = random.uniform(*radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))

    def color_jitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        """
        Zmiana jasności, kontrastu i nasycenia kolorów w obrazie.

        Args:
            img: Obraz wejściowy.
            brightness (float): Maksymalne odchylenie zmiany jasności.
            contrast (float): Maksymalne odchylenie kontrastu.
            saturation (float): Maksymalne odchylenie nasycenia.
            hue (float): Maksymalne odchylenie barwy.

        Returns:
            PIL.Image : Obraz po zmodyfikowaniu kolorystyki.
        """
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(img)

    def add_gaussian_noise(self, img, mean=0, std=0.01):
        """
        Dodanie szumu do obrazu.

        Args:
            img: Obraz wejściowy.
            std (float): Odchylenie określające intensywność szumu.

        Returns:
            PIL.Image: Obraz po dodaniu szumu.
        """
        img_tensor = F.to_tensor(img)
        noise = torch.randn_like(img_tensor) * std
        noisy_img = img_tensor + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        return F.to_pil_image(noisy_img)