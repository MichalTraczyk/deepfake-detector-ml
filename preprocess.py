import os
import random
import shutil

import cv2

from FaceProcessor import ImageFaceProcessor

image_processor = ImageFaceProcessor((256, 256))


def process_image(image):
    global image_processor
    face = image_processor.get_face(image)
    return face


def split_images_to_train_val_test():
    input_dir = 'data_processed'
    output_dir = 'data_split'
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    classes = ['real', 'fake']
    random.seed(42)

    # Create output folders
    for split in split_ratios:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in classes:
        src_dir = os.path.join(input_dir, cls)
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
                dst_path = os.path.join(output_dir, split_name, cls, file)
                shutil.move(src_path, dst_path)

    print("✅ Files moved and split into train/val/test.")


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video{video_path}")
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
    data_common_path = "data/Celeb-DF-v2/"
    real_paths = [
        data_common_path + "Celeb-real",
        data_common_path + "YouTube-real"
    ]
    fake_paths = [
        data_common_path + "Celeb-synthesis"
    ]
    for videos_path in real_paths:
        files_list = os.listdir(videos_path)
        processed = 0
        for video in files_list:
            video_name = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(save_directory, "real", video_name)
            video_path = os.path.join(videos_path, video)
            (count, saved) = process_video(video_path, path)
            print(f"real: {round(processed / len(files_list), 2)} Processed {video}: saved {saved}/{count} frames.")
            processed += 1

    for videos_path in fake_paths:
        files_list = os.listdir(videos_path)
        processed = 0
        for video in files_list:
            video_name = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(save_directory, "fake", video_name)
            video_path = os.path.join(videos_path, video)
            (count, saved) = process_video(video_path, path)
            print(f"fake: {round(processed / len(files_list), 2)} Processed {video}: saved {saved}/{count} frames.")
            processed += 1
    split_images_to_train_val_test()
