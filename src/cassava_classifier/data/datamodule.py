from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomResizedCrop,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import CassavaDataset


def get_transforms(img_size, is_train=True):
    if is_train:
        return Compose(
            [
                RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )
    else:
        return Compose(
            [
                Resize(height=img_size, width=img_size),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )


class CassavaDataModule(LightningDataModule):
    def __init__(
        self, train_df, val_df, model_config, dataroot, batch_size=32, num_workers=4
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.model_config = model_config
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CassavaDataset(
                self.train_df,
                self.dataroot,
                transform=get_transforms(self.model_config["img_size"], is_train=True),
                divide_image=self.model_config.get("divide_image", False),
                img_size=self.model_config["img_size"],
            )
            self.val_dataset = CassavaDataset(
                self.val_df,
                self.dataroot,
                transform=get_transforms(self.model_config["img_size"], is_train=False),
                divide_image=self.model_config.get("divide_image", False),
                img_size=self.model_config["img_size"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
