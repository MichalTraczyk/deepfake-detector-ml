import cv2
import face_recognition
import numpy as np


class ImageFaceProcessor:
    def __init__(self, resolution):
        self.resolution = resolution

    def is_blurry(self, image, lap_threshold=3.0, ten_threshold=15):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Sobel-based (Tenengrad) blur detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        tenengrad_score = np.mean(sobel_magnitude)

        # cv2.putText(image, f"Lap: {laplacian_var:.1f}", (10, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # cv2.putText(image, f"Ten: {tenengrad_score:.1f}", (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return laplacian_var < lap_threshold or tenengrad_score < ten_threshold

    def get_face(self, image):
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return None
        top, right, bottom, left = face_locations[0]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, self.resolution)
        if self.is_blurry(face):
            return None
        return face
