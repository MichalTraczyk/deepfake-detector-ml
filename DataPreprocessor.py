import os

import cv2

from FaceProcessor import ImageFaceProcessor


class DataPreprocessor:

    def __init__(self, source_directory: str, save_directory: str):
        self.save_directory = save_directory
        self.source_directory = source_directory
        self.image_processor = ImageFaceProcessor((256, 256))

    def process_directory(self):
        all_files = len(os.listdir(self.source_directory))
        processed_count = 0
        failed = 0
        for filename in os.listdir(self.source_directory):
            full_path = os.path.join(self.source_directory, filename)
            img = cv2.imread(full_path)
            processed = self.process_image(img)
            save_dir = os.path.join(self.save_directory, filename)
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
            if processed is None:
                failed += 1
                continue
            success = cv2.imwrite(save_dir, processed)
            if not success:
                failed += 1
                print("error saving file: ", save_dir)
            processed_count += 1
            print("Processed: " + str(processed_count / all_files), " from: ", self.source_directory)
        print("failed: ", failed)

    def process_image(self, image):
        face = self.image_processor.get_face(image)
        return face

if __name__ == "__main__":
    dirs_to_process = [
         "data/Test/Fake/",
         "data/Test/Real/",
         "data/Train/Fake/",
         "data/Train/Real/",
         "data/Validation/Fake",
         "data/Validation/Real"
    ]

    for dir in dirs_to_process:
        print("Processing: ", dir)
        processor = DataPreprocessor(dir, dir.replace("data", "data_processed"))
        processor.process_directory()