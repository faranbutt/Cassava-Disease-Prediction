# working/Cassava-Disease-Detection/src/cassava_classifier/utils/preprocessing.py
from pathlib import Path

import cv2
import pandas as pd


def clean_labels(df, data_path: str):
    """Remove corrupted/invalid images"""
    cleaned = []
    for img, label in zip(df["image_id"], df["label"]):
        path = Path(data_path) / "train_images" / img
        im = cv2.imread(str(path))
        if im is None or im.mean() < 5:
            continue
        cleaned.append([img, label])
    cleaned_df = pd.DataFrame(cleaned, columns=["image_id", "label"])
    print(f"Cleaned: {len(df)} → {len(cleaned_df)} samples")
    return cleaned_df
