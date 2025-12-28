import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from timm import create_model
from torchmetrics import Accuracy

from .components import (
    AttentionWeighting,
    LabelSmoothingCrossEntropy,
    MultiDropoutLinear,
)


class CassavaLightningModule(pl.LightningModule):
    def __init__(self, model_config, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.lr = lr

        pretrained = model_config.get("pretrained", True)
        base_model = create_model(model_config["name"], pretrained=pretrained)
        n_features = base_model.head.in_features
        base_model.head = nn.Identity()
        self.backbone = base_model

        self.use_attention = model_config.get("use_attention", False)
        self.divide_image = model_config.get("divide_image", False)
        if self.use_attention and self.divide_image:
            pattern = model_config.get("attention_pattern", "A")
            self.attention = AttentionWeighting(n_features, pattern=pattern)

        if model_config.get("use_multidrop", False):
            self.head = MultiDropoutLinear(n_features, 5, n_drops=5, dropout_rate=0.5)
        else:
            self.head = nn.Linear(n_features, 5)

        label_smoothing = model_config.get("label_smoothing", 0.0)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        else:
            weights = torch.tensor([1.0, 1.0, 1.0, 0.25, 1.0])
            self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.accuracy = Accuracy(task="multiclass", num_classes=5)
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        if self.divide_image:
            B, C, H, W = x.shape
            assert H == W == 448, f"Expected 448x448, got {H}x{W}"
            parts = [
                x[:, :, :224, :224],
                x[:, :, :224, 224:],
                x[:, :, 224:, :224],
                x[:, :, 224:, 224:],
            ]
            features_list = [self.backbone(part) for part in parts]
            if self.use_attention:
                features = self.attention(features_list)
            else:
                features = torch.stack(features_list).mean(dim=0)
        else:
            features = self.backbone(x)
        return self.head(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.train_preds.extend(preds.cpu().numpy())
        self.train_targets.extend(y.cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        if self.train_preds:
            f1 = f1_score(
                self.train_targets,
                self.train_preds,
                average="weighted",
                zero_division=0,
            )
            self.log("train_f1", f1)
            self.train_preds.clear()
            self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        self.val_preds.clear()
        self.val_targets.clear()
