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

def get_processed_video_names(dir_path):
    names = set()
    for filename in os.listdir(dir_path):
        if "_frame_" in filename:
            base_name = filename.split("_frame_")[0]
            names.add(base_name)
    return names


def split_images_to_dirs():
    base_dir = 'data/02_processed'
    temp_dirs = {
        'real': os.path.join(base_dir, 'real'),
        'fake': os.path.join(base_dir, 'fake')
    }

    target_train_ratio = 0.8
    classes = ['real', 'fake']

    random.seed(42)

    for split in ['train', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    for cls in classes:
        src_dir = temp_dirs[cls]
        if not os.path.exists(src_dir):
            continue

        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        video_groups = {}

        for f in files:
            if "_frame_" in f:
                video_id = f.split("_frame_")[0]
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append(f)

        total_frames_in_class = len(files)
        target_train_frames = int(total_frames_in_class * target_train_ratio)

        video_ids = list(video_groups.keys())
        random.shuffle(video_ids)

        train_videos = []
        test_videos = []

        current_train_frames = 0

        for vid in video_ids:
            frames_in_video = len(video_groups[vid])

            if current_train_frames < target_train_frames:
                train_videos.append(vid)
                current_train_frames += frames_in_video
            else:
                test_videos.append(vid)

        actual_train_ratio = current_train_frames / total_frames_in_class
        print(f"[{cls.upper()}] Total Frames: {total_frames_in_class}")
        print(f"   - Target Split: 80% / 20%")
        print(f"   - Actual Split: {actual_train_ratio:.2%} / {(1 - actual_train_ratio):.2%}")
        print(f"   - Train Videos: {len(train_videos)} | Test Videos: {len(test_videos)}")

        for vid in train_videos:
            for file in video_groups[vid]:
                shutil.move(os.path.join(src_dir, file), os.path.join(base_dir, 'train', cls, file))

        for vid in test_videos:
            for file in video_groups[vid]:
                shutil.move(os.path.join(src_dir, file), os.path.join(base_dir, 'test', cls, file))

        # Cleanup
        try:
            os.rmdir(src_dir)
        except:
            pass

    print("✅ Split complete. Balanced by Frame Count, Grouped by Video ID.")


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print(f"Error: Could not open video{video_path}")
        return
    frame_count = 0
    saved_count = 0
    skipped_in_row = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            img = process_image(frame)
            if img is None:
                skipped_in_row += 1
                if skipped_in_row == 60:
                    break
                continue
            skipped_in_row = 0
            filename = f"{output_path}_frame_{saved_count:05d}.jpg"
            cv2.imwrite(filename, img)
            saved_count += 1

        frame_count += 1
    cap.release()
    return frame_count, saved_count


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Usibng gpu!")
    save_directory = "data/02_processed/"
    os.makedirs(save_directory, exist_ok=True)

    data_common_path = "data/01_raw/Celeb-DF-v2/"
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
        files_list.sort()
        total_videos = len(files_list)
        output_dir = real_save_path

        with tqdm(total=total_videos, desc=f"Processing real videos from {videos_path}") as pbar:
            i = 0
            while i < total_videos:
                video = files_list[i]
                video_name = os.path.splitext(os.path.basename(video))[0]

                next_video_name = None
                if i + 1 < total_videos:
                    next_video = files_list[i + 1]
                    next_video_name = os.path.splitext(os.path.basename(next_video))[0]

                if video_name in processed_real or (next_video_name and next_video_name in processed_real):
                    pbar.update(2)
                    i += 2
                    continue
                video_path = os.path.join(videos_path, video)
                path = os.path.join(output_dir, video_name)
                (count, saved) = process_video(video_path, path)
                pbar.update(1)
                i += 1

    for videos_path in fake_paths:
        files_list = os.listdir(videos_path)
        files_list.sort()
        total_videos = len(files_list)
        output_dir = fake_save_path

        with tqdm(total=total_videos, desc=f"Processing fake videos from {videos_path}") as pbar:
            i = 0
            while i < total_videos:
                video = files_list[i]
                video_name = os.path.splitext(os.path.basename(video))[0]

                next_video_name = None
                if i + 1 < total_videos:
                    next_video = files_list[i + 1]
                    next_video_name = os.path.splitext(os.path.basename(next_video))[0]

                if video_name in processed_fake or (next_video_name and next_video_name in processed_fake):
                    pbar.update(2)
                    i += 2
                    continue
                video_path = os.path.join(videos_path, video)
                path = os.path.join(output_dir, video_name)
                (count, saved) = process_video(video_path, path)
                pbar.update(1)
                i += 1

    split_images_to_dirs()
