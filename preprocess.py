import os

import cv2

from FaceProcessor import ImageFaceProcessor

image_processor = ImageFaceProcessor((256, 256))


def process_image(image):
    global image_processor
    face = image_processor.get_face(image)
    return face


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
            print(f"real: {round(processed/len(files_list),2)} Processed {video}: saved {saved}/{count} frames.")
            processed+=1

    for videos_path in fake_paths:
        files_list = os.listdir(videos_path)
        processed = 0
        for video in files_list:
            video_name = os.path.splitext(os.path.basename(video))[0]
            path = os.path.join(save_directory, "fake", video_name)
            video_path = os.path.join(videos_path, video)
            (count, saved) = process_video(video_path, path)
            print(f"fake: {round(processed/len(files_list),2)} Processed {video}: saved {saved}/{count} frames.")
            processed+=1