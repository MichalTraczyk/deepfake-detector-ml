import os

import cv2

from FaceProcessor import ImageFaceProcessor


class DataPreprocessor:

    def __init__(self, source_directory: str, save_directory: str):
        self.save_directory = save_directory
        self.source_directory = source_directory
        self.image_processor = ImageFaceProcessor((256, 256))

    def process_directory(self):
        for filename in os.listdir(self.source_directory):
            print(self.source_directory + filename)
            full_path = os.path.join(self.source_directory, filename)
            img = cv2.imread(full_path)
            processed = self.process_image(img)
            save_dir = os.path.join(self.save_directory, filename)
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            cv2.imwrite(save_dir, processed)

    def process_image(self, image):
        face = self.image_processor.get_face(image)
        return face
