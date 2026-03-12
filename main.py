import argparse
import sys
import os

from src.data_preparation import prepare_data
from src.train import train_pipeline
from src.predict import predict_video

def main():
    parser = argparse.ArgumentParser(description="UCF101 Video Classification Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare data
    subparsers.add_parser("prepare", help="Download and prepare the dataset")

    # Train
    subparsers.add_parser("train", help="Train the sequence model")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict class of a single video")
    predict_parser.add_argument("video_path", type=str, help="Path to the video file")
    predict_parser.add_argument("--model_path", type=str, default="models/final_model", help="Path to the trained model")

    # Full pipeline
    subparsers.add_parser("full", help="Run the full pipeline (prepare + train)")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data()
    elif args.command == "train":
        train_pipeline()
    elif args.command == "predict":
        if not os.path.exists(args.video_path):
            print(f"Error: Video file {args.video_path} not found.")
            sys.exit(1)
        if not os.path.exists(args.model_path):
            print(f"Error: Model {args.model_path} not found. Did you run training?")
            sys.exit(1)
        predict_video(args.video_path, args.model_path)
    elif args.command == "full":
        prepare_data()
        train_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
