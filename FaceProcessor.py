import cv2

class ImageFaceProcessor:
    def __init__(self, resolution):
        self.resolution = resolution
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]  # take first face
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, self.resolution)
        return face