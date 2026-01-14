import configparser
import os
import random
import shutil
from tqdm import tqdm
import torch
import cv2

from FaceProcessor import ImageFaceProcessor

config_override = configparser.ConfigParser()
config_override.read('config.ini')

res = (int)(config_override["LearningSettings"]["ImageResolution"])
image_processor = ImageFaceProcessor((res, res))


def process_image(image):
    global image_processor
    face = image_processor.get_face(image)
    return face
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            img = process_image(frame)
            if img is not None:
                filename = f"{output_path}.jpg"
                cv2.imwrite(filename, img)
                break
        frame_count += 1
    cap.release()

def process_path(source_path:str,target_path:str):
    files_list = os.listdir(source_path)
    total_videos = len(files_list)
    with tqdm(total=total_videos, desc=f"Processing videos from {source_path}") as pbar:
        i = 0
        while i < total_videos:
            video = files_list[i]
            video_name = os.path.splitext(os.path.basename(video))[0]
            video_path = os.path.join(source_path, video)
            path = os.path.join(target_path, video_name)
            process_video(video_path, path)
            pbar.update(1)
            i += 1


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using gpu!")
    save_directory = "data/face_forentics_processed/"
    os.makedirs(save_directory, exist_ok=True)

    data_common_path = "data/face_forentics/"
    real_path = data_common_path + "original/"
    fake_path = data_common_path + "FaceShifter/"

    real_save_path = os.path.join(save_directory, "real")
    fake_save_path = os.path.join(save_directory, "fake")

    os.makedirs(real_save_path, exist_ok=True)
    os.makedirs(fake_save_path, exist_ok=True)

    process_path(real_path, real_save_path)
    process_path(fake_path, fake_save_path)


