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
        features = feature_extractor.predict(frames[:length], verbose=0)
        frame_features[0, :length, :] = features
        frame_mask[0, :length] = 1

    return frame_features, frame_mask

def predict_video(video_path, model_path, feature_extractor=None):
    print(f"🎬 Loading video: {os.path.basename(video_path)}...")
    frames = load_video(video_path)
    if len(frames) == 0:
        print("❌ Error: Could not load any frames from the video.")
        return []

    print("🤖 Initializing AI models (this may take a moment)...")
    if feature_extractor is None:
        feature_extractor = build_feature_extractor()
    model = keras.models.load_model(model_path)

    print("🧠 Extracting features...")
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)

    print("✨ Classifying action...")
    probabilities = model.predict([frame_features, frame_mask], verbose=0)[0]

    print("\n" + "═" * 48)
    print(f"{'ACTION CLASS':<20} {'CONFIDENCE':<25}")
    print("─" * 48)

    results = []
    sorted_indices = np.argsort(probabilities)[::-1]
    for idx, i in enumerate(sorted_indices):
        prob = probabilities[i]
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        label = CLASSES[i]

        if idx == 0: # Highlight top prediction
            print(f"\033[1;32m{label:<20}\033[0m {bar} {prob * 100:5.2f}%")
        else:
            print(f"{label:<20} {bar} {prob * 100:5.2f}%")
        results.append({"class": label, "probability": prob})

    print("═" * 48)

    top_prob = probabilities[sorted_indices[0]]
    top_label = CLASSES[sorted_indices[0]]

    if top_prob >= 0.8:
        status, color = "✅ Confident", "\033[1;32m" # Green
    elif top_prob >= 0.5:
        status, color = "⚠️  Uncertain", "\033[1;33m" # Yellow
    else:
        status, color = "❓ Low Confidence", "\033[1;31m" # Red

    print(f"{status}: {color}{top_label}\033[0m ({top_prob * 100:.1f}%)\n")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <video_path> <model_path>")
    else:
        predict_video(sys.argv[1], sys.argv[2])
