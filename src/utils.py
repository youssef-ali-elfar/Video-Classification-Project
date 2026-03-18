import cv2
import numpy as np
import os
import imageio
from .config import IMG_SIZE

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            # BGR to RGB
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def save_as_gif(frames, output_path, fps=10):
    """Saves a sequence of frames as a GIF file."""
    converted_frames = frames.astype(np.uint8)
    imageio.mimsave(output_path, converted_frames, fps=fps)
    print(f"Saved animation to {output_path}")

def get_video_info(path):
    """Returns FPS, frame count, and duration of a video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, frame_count, duration

def print_header(title):
    """Prints a styled ASCII header for CLI sections."""
    width = 50
    print("\n" + "═" * width)
    print(f" {title.upper()} ".center(width, "■"))
    print("═" * width + "\n")
