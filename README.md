# UCF101 Video Classification Project

This project implements a deep learning pipeline for human action recognition using the UCF101 dataset. It uses a hybrid approach:
1. **Feature Extraction**: Uses a pretrained InceptionV3 model to extract spatial features from video frames.
2. **Sequence Modeling**: Uses an LSTM/GRU-based RNN to model the temporal dependencies between frames and classify the action.

## Project Structure

```
├── main.py              # Entry point for the pipeline
├── src/                 # Source code
│   ├── config.py        # Configuration and hyperparameters
│   ├── data_preparation.py # Dataset downloading and processing
│   ├── feature_extraction.py # Feature extraction logic
│   ├── model.py         # RNN model architecture
│   ├── train.py         # Training pipeline
│   ├── predict.py       # Inference logic
│   └── utils.py         # Video processing utilities
├── models/              # Directory for saved models (ignored by git)
├── data/                # Directory for dataset (ignored by git)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**:
   This will download the videos for the selected classes and create the training/test splits.
   ```bash
   python main.py prepare
   ```

## Usage

### Training

To train the model from scratch:
```bash
python main.py train
```
The best model will be saved in `models/final_model`.

### Inference

To predict the action in a single video:
```bash
python main.py predict path/to/video.avi
```

### Full Pipeline

To run both preparation and training:
```bash
python main.py full
```

## Configuration

You can modify classes, hyperparameters, and paths in `src/config.py`.

Default classes:
- HandstandPushups
- HandstandWalking
- PullUps
- Punch
- PushUps

## Author
**Youssef Elfar**
Refactored and modularized for better maintainability.
