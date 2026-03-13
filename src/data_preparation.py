import os
import ssl
import re
import urllib.request
import pandas as pd
import numpy as np
from .config import UCF_ROOT, CLASSES, DATA_ROOT, MODELS_ROOT

def list_ucf_videos():
    unverified_context = ssl._create_unverified_context()
    index = urllib.request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")

    videos_dict = {}
    for classname in CLASSES:
        s = "(v_" + str(classname) + "[\\w_]+\\.avi)"
        videos = re.findall(s, index)
        videos_dict[classname] = sorted(set(videos))
    return videos_dict

def download_videos(videos_dict):
    unverified_context = ssl._create_unverified_context()
    total_videos = sum(len(videos) for videos in videos_dict.values())
    count = 0

    for classname, video_list in videos_dict.items():
        class_dir = os.path.join(DATA_ROOT, classname)
        os.makedirs(class_dir, exist_ok=True)

        for video in video_list:
            count += 1
            video_path = os.path.join(class_dir, video)
            if not os.path.exists(video_path):
                url = UCF_ROOT + video
                print(f"[{count}/{total_videos}] Downloading {url}...")
                try:
                    data = urllib.request.urlopen(url, context=unverified_context).read()
                    with open(video_path, "wb") as f:
                        f.write(data)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            else:
                # Still show progress even if file exists
                if count % 10 == 0 or count == total_videos:
                    print(f"[{count}/{total_videos}] Skipping {video} (already exists)")

def create_dataset_csv():
    data = []
    for classname in CLASSES:
        class_dir = os.path.join(DATA_ROOT, classname)
        if not os.path.exists(class_dir):
            continue
        for video_file in os.listdir(class_dir):
            if video_file.endswith(".avi"):
                data.append([CLASSES.index(classname), os.path.join(class_dir, video_file)])

    df = pd.DataFrame(data, columns=["class", "path"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def split_and_save_dataset(df):
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_df.to_pickle(os.path.join(DATA_ROOT, 'train.pkl'))
    test_df.to_pickle(os.path.join(DATA_ROOT, 'test.pkl'))
    print(f"Saved train and test pickles to {DATA_ROOT}")

def prepare_data():
    print("Listing videos...")
    videos_dict = list_ucf_videos()
    print("Downloading videos (this may take a while)...")
    download_videos(videos_dict)
    print("Creating dataset CSV...")
    df = create_dataset_csv()
    print("Splitting and saving dataset...")
    split_and_save_dataset(df)

if __name__ == "__main__":
    prepare_data()
