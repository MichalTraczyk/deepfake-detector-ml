import os
import random
import shutil
import cv2
from tqdm import tqdm


def process_video_to_frames(video_path, output_path, image_processor, frames_per_video=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    frame_count = 0
    saved_count = 0
    skipped_in_row = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:
            img = image_processor.get_face(frame)
            if img is None:
                skipped_in_row += 1
                if skipped_in_row == 60: break
                continue

            skipped_in_row = 0
            #Celeb-DF
            if frames_per_video is None:
                filename = f"{output_path}_frame_{saved_count:05d}.jpg"
            else:
                #FaceForensics style
                filename = f"{output_path}.jpg"

            cv2.imwrite(filename, img)
            saved_count += 1

            if frames_per_video and saved_count >= frames_per_video:
                break

        frame_count += 1
    cap.release()
    return frame_count, saved_count


def run_extraction(raw_dirs, target_base_dir, res, mode="multi"):
    from deepfake_detector.utils.FaceProcessor import ImageFaceProcessor
    processor = ImageFaceProcessor((res, res))

    for label, paths in raw_dirs.items():
        save_path = os.path.join(target_base_dir, label)
        os.makedirs(save_path, exist_ok=True)

        # Get list of already processed files to support continuation
        existing_files = set(os.listdir(save_path))

        for folder in paths:
            files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi'))]
            for video in tqdm(files, desc=f"Processing {label}"):
                video_name = os.path.splitext(video)[0]
                if any(video_name in f for f in existing_files):
                    continue

                v_path = os.path.join(folder, video)
                out_path = os.path.join(save_path, video_name)
                limit = 1 if mode == "single" else None
                process_video_to_frames(v_path, out_path, processor, limit)

    return target_base_dir


def split_data(processed_dir, train_ratio=0.8):
    classes = ['real', 'fake']
    random.seed(42)

    for cls in classes:
        src_dir = os.path.join(processed_dir, cls)
        if not os.path.exists(src_dir): continue

        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        video_groups = {}

        for f in files:
            video_id = f.split("_frame_")[0] if "_frame_" in f else os.path.splitext(f)[0]
            video_groups.setdefault(video_id, []).append(f)

        video_ids = list(video_groups.keys())
        random.shuffle(video_ids)

        # Logic to split based on frame count
        total_frames = len(files)
        target_train = int(total_frames * train_ratio)
        current_train = 0

        for vid in video_ids:
            split = 'train' if current_train < target_train else 'test'
            dest_dir = os.path.join(processed_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for file in video_groups[vid]:
                shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))

            if split == 'train': current_train += len(video_groups[vid])

        try:
            os.rmdir(src_dir)
        except:
            pass

    return "Split Complete"