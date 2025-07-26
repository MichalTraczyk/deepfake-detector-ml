import cv2
import face_recognition
import numpy as np
import face_alignment
import time
import torch

class ImageFaceProcessor:
    def __init__(self, resolution):
        self.resolution = resolution
        self.aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=('cuda' if torch.cuda.is_available() else 'cpu'))

    def is_blurry(self, image, lap_threshold=3.0, ten_threshold=15):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        tenengrad_score = np.mean(sobel_magnitude)
        return laplacian_var < lap_threshold or tenengrad_score < ten_threshold

    def align_face(self, image, landmarks):
        # Align based on eye centers
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)

        rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        aligned = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return aligned
    def get_face(self, image):
        preds = self.aligner.get_landmarks(image)
        if preds is None:
            return None
        landmarks = preds[0]
        aligned_image = self.align_face(image, landmarks)

        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            return None
        top, right, bottom, left = face_locations[0]
        face = aligned_image[top:bottom, left:right]
        face = cv2.resize(face, self.resolution)
        if self.is_blurry(face):
            return None
        return face
