import configparser
import os
import random
import shutil
from tqdm import tqdm

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
def get_processed_video_names(dir_path):
    """Return set of base video names (without _frame_*) from saved files."""
    names = set()
    for filename in os.listdir(dir_path):
        if "_frame_" in filename:
            base_name = filename.split("_frame_")[0]
            names.add(base_name)
    return names

def split_images_to_train_val_test():
    base_dir = 'data_processed'
    temp_dirs = {
        'real': os.path.join(base_dir, 'real'),
        'fake': os.path.join(base_dir, 'fake')
    }
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    classes = ['real', 'fake']
    random.seed(42)

    # Create output folders inside base_dir
    for split in split_ratios:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    for cls in classes:
        src_dir = temp_dirs[cls]
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        random.shuffle(files)

        total = len(files)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])

        splits = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        for split_name, split_files in splits.items():
            for file in split_files:
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(base_dir, split_name, cls, file)
                shutil.move(src_path, dst_path)

        # Remove the empty temp class directory
        os.rmdir(src_dir)

    print("✅ Files split into train/val/test inside 'data_processed/'.")


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print(f"Error: Could not open video{video_path}")
        return
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            img = process_image(frame)
            if img is None:
                continue
            filename = f"{output_path}_frame_{saved_count:05d}.jpg"
            cv2.imwrite(filename, img)
            saved_count += 1

        frame_count += 1
    cap.release()
    return frame_count, saved_count


if __name__ == "__main__":
    save_directory = "data_processed/"
    os.makedirs(save_directory, exist_ok=True)

    data_common_path = "data/Celeb-DF-v2/"
    real_paths = [
        data_common_path + "Celeb-real",
        data_common_path + "YouTube-real"
    ]
    fake_paths = [
        data_common_path + "Celeb-synthesis"
    ]

    real_save_path = os.path.join(save_directory, "real")
    fake_save_path = os.path.join(save_directory, "fake")

    os.makedirs(real_save_path, exist_ok=True)
    os.makedirs(fake_save_path, exist_ok=True)

    processed_real = get_processed_video_names(real_save_path)
    processed_fake = get_processed_video_names(fake_save_path)

    for videos_path in real_paths:
        files_list = os.listdir(videos_path)
        total_videos = len(files_list)
        output_dir = real_save_path

        with tqdm(total=total_videos, desc=f"Processing real videos from {videos_path}") as pbar:
            for video in files_list:
                video_name = os.path.splitext(os.path.basename(video))[0]
                path = os.path.join(output_dir, video_name)

                if video_name in processed_real:
                    pbar.update(1)
                    continue

                video_path = os.path.join(videos_path, video)
                (count, saved) = process_video(video_path, path)
                #pbar.write(f"📹 Processed {video}: saved {saved}/{count} frames.")
                pbar.update(1)

    # Process fake videos
    for videos_path in fake_paths:
        files_list = os.listdir(videos_path)
        total_videos = len(files_list)
        output_dir = fake_save_path

        with tqdm(total=total_videos, desc=f"Processing fake videos from {videos_path}") as pbar:
            for video in files_list:
                video_name = os.path.splitext(os.path.basename(video))[0]
                path = os.path.join(output_dir, video_name)

                if video_name in processed_fake:
                    pbar.update(1)
                    continue

                video_path = os.path.join(videos_path, video)
                (count, saved) = process_video(video_path, path)
                #pbar.write(f"📹 Processed {video}: saved {saved}/{count} frames.")
                pbar.update(1)
    split_images_to_train_val_test()
