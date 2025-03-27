import cv2
import face_recognition


class ImageFaceProcessor:

    def __init__(self, resolution):
        self.resolution = resolution

    def get_face(self, image):
        rgb_image = image[:, :, ::-1]  # bgr to rgb
        face_locations = face_recognition.face_locations(rgb_image)
        face_image = None
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_image = image[top:bottom, left:right]
            face_image = cv2.resize(face_image, self.resolution)
        return face_image
