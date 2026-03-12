import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .config import MAX_SEQ_LENGTH, NUM_FEATURES, CLASSES
from .utils import load_video
from .feature_extraction import build_feature_extractor

def prepare_single_video(frames, feature_extractor):
    video_length = frames.shape[0]
    length = min(MAX_SEQ_LENGTH, video_length)

    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    if length > 0:
        features = feature_extractor.predict(frames[:length])
        frame_features[0, :length, :] = features
        frame_mask[0, :length] = 1

    return frame_features, frame_mask

def predict_video(video_path, model_path, feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = build_feature_extractor()

    model = keras.models.load_model(model_path)

    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)

    probabilities = model.predict([frame_features, frame_mask])[0]

    # We assume the model was trained with CLASSES in config
    results = []
    for i in np.argsort(probabilities)[::-1]:
        results.append({
            "class": CLASSES[i],
            "probability": probabilities[i]
        })
        print(f"  {CLASSES[i]}: {probabilities[i] * 100:5.2f}%")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <video_path> <model_path>")
    else:
        predict_video(sys.argv[1], sys.argv[2])
