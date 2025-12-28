from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    def __init__(self, df, data_path, transform=None, divide_image=False, img_size=384):
        self.df = df
        self.data_path = Path(data_path)
        self.img_ids = df["image_id"].values
        self.labels = df["label"].values if "label" in df.columns else None
        self.transform = transform
        self.divide_image = divide_image
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.data_path / "train_images" / img_id
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image
