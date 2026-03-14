import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc

from .config import DATA_ROOT, MODELS_ROOT, MAX_SEQ_LENGTH, NUM_FEATURES, CLASSES, EPOCHS
from .utils import load_video
from .feature_extraction import build_feature_extractor
from .model import get_sequence_model

def prepare_all_videos(df, feature_extractor, label_processor):
    num_samples = len(df)
    video_paths = df["path"].values.tolist()
    labels = df["class"].values
    labels = label_processor(labels[..., None]).numpy()

    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    for idx, path in enumerate(video_paths):
        frames = load_video(path)

        # Corrected bug: video_length is frames.shape[0]
        video_length = frames.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)

        if length > 0:
            # Optimize: predict on all frames at once
            features = feature_extractor.predict(frames[:length], verbose=0)
            frame_features[idx, :length, :] = features
            frame_masks[idx, :length] = 1

        percent = (idx + 1) / num_samples * 100
        print(f"\rProcessed {idx + 1}/{num_samples} videos ({percent:5.1f}%)", end="", flush=True)
        if idx % 50 == 0:
            gc.collect()

    print() # New line after progress

    return (frame_features, frame_masks), labels

def train_pipeline():
    # Load dataset info
    train_df = pd.read_pickle(os.path.join(DATA_ROOT, 'train.pkl'))
    test_df = pd.read_pickle(os.path.join(DATA_ROOT, 'test.pkl'))

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df["class"])
    )

    feature_extractor = build_feature_extractor()

    print("Preparing training data...")
    train_data, train_labels = prepare_all_videos(train_df, feature_extractor, label_processor)
    print("Preparing test data...")
    test_data, test_labels = prepare_all_videos(test_df, feature_extractor, label_processor)

    checkpoint_path = os.path.join(MODELS_ROOT, "video_classifier_checkpoint")
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1
    )

    num_classes = len(label_processor.get_vocabulary())
    seq_model = get_sequence_model(num_classes)

    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(checkpoint_path)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    model_save_path = os.path.join(MODELS_ROOT, "final_model")
    seq_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return history

if __name__ == "__main__":
    train_pipeline()
