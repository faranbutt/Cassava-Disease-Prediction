# src/cassava_classifier/pipelines/train.py
import logging
import os
import subprocess
import sys
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold

from cassava_classifier.data.datamodule import CassavaDataModule
from cassava_classifier.models.model import CassavaLightningModule
from cassava_classifier.pipelines.convert import convert_to_onnx
from cassava_classifier.pipelines.infer import ensemble_predict
from cassava_classifier.utils.preprocessing import clean_labels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

torch.serialization.add_safe_globals([typing.Any])
log = logging.getLogger(__name__)


class MetricsHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.train_loss.append(
            metrics.get("train_loss_epoch", torch.tensor(0.0)).item()
        )
        self.val_loss.append(metrics.get("val_loss", torch.tensor(0.0)).item())
        self.train_acc.append(metrics.get("train_acc_epoch", torch.tensor(0.0)).item())
        self.val_acc.append(metrics.get("val_acc", torch.tensor(0.0)).item())


def train_fold(fold: int, train_df, val_df, model_config: DictConfig, cfg: DictConfig):
    tracking_uri = cfg.train.mlflow.get("tracking_uri", "sqlite:///mlflow.db")
    artifact_location = cfg.train.mlflow.get("artifact_location", "./mlartifacts")
    experiment_name = cfg.train.mlflow.get(
        "experiment_name", "cassava-disease-classification"
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name, artifact_location=artifact_location
        )
    mlflow.set_experiment(experiment_name)
    run_name = f"{cfg.model_name}_fold_{fold}"
    data_module = CassavaDataModule(
        train_df,
        val_df,
        model_config=model_config,
        dataroot=cfg.data.dataroot,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    model = CassavaLightningModule(model_config=model_config, lr=cfg.train.lr)
    checkpoint_dir = Path(cfg.data.output_dir) / "models" / cfg.model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    history_cb = MetricsHistory()
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=tracking_uri, run_name=run_name
    )
    log_every = min(
        cfg.train.log_every_n_steps, max(1, len(train_df) // cfg.train.batch_size)
    )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        precision=cfg.train.precision,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=cfg.train.patience),
            LearningRateMonitor(),
            history_cb,
        ],
        log_every_n_steps=log_every,
        enable_progress_bar=True,
        logger=mlflow_logger,
    )
    trainer.fit(model, data_module)
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        raise FileNotFoundError(f"No checkpoint saved for fold {fold}")
    val_metrics = trainer.callback_metrics or {}
    train_metrics = trainer.logged_metrics or {}
    mlflow.log_metrics(
        {
            "val_loss": float(val_metrics.get("val_loss", 0)),
            "val_acc": float(val_metrics.get("val_acc", 0)),
            "train_loss": float(train_metrics.get("train_loss_epoch", 0)),
            "train_acc": float(train_metrics.get("train_acc_epoch", 0)),
        }
    )
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"metrics_fold_{fold}_{cfg.model_name}.png"
    try:
        create_metrics_plot(history_cb, str(plot_path))
        mlflow.log_artifact(str(plot_path), artifact_path="plots")
    except Exception as e:
        print(f"Opps: couldnt create/log metrics plot: {e}")
    try:
        mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
        best_model = CassavaLightningModule.load_from_checkpoint(
            best_ckpt_path, model_config=model_config
        )
        registered_model_name = f"{experiment_name}-{cfg.model_name}"
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )
    except Exception as e:
        print(f"warning: failed to log/register model to MLflow: {e}")
    return best_ckpt_path


def create_metrics_plot(history_cb, plot_path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    epochs = list(range(1, len(history_cb.train_loss) + 1))
    ax[0, 0].plot(epochs, history_cb.train_loss, "b-", label="Train Loss")
    ax[0, 0].set_title("Training Loss")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].legend()
    ax[0, 1].plot(epochs, history_cb.val_loss, "r-", label="Val Loss")
    ax[0, 1].set_title("Validation Loss")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("Loss")
    ax[0, 1].legend()
    ax[1, 0].plot(epochs, history_cb.train_acc, "b-", label="Train Acc")
    ax[1, 0].set_title("Training Accuracy")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].legend()
    ax[1, 1].plot(epochs, history_cb.val_acc, "r-", label="Val Acc")
    ax[1, 1].set_title("Validation Accuracy")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def train_model(cfg: DictConfig):
    model_config = cfg.model
    pl.seed_everything(cfg.train.seed, workers=True)

    df = pd.read_csv(Path(cfg.data.dataroot) / "train.csv")
    df = clean_labels(df, cfg.data.dataroot)

    if cfg.get("debug", False):
        df = df.sample(
            n=min(cfg.debug_samples, len(df)), random_state=cfg.train.seed
        ).reset_index(drop=True)
        print(f"DEBUG MODE: Using {len(df)} samples")

    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        best_path = train_fold(fold, train_df, val_df, model_config, cfg)
        print(f"Fold {fold} best model: {best_path}")


def get_best_checkpoint(model_dir: Path) -> Path:
    checkpoint = model_dir / "model_best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    return checkpoint


def cleanup_fold_checkpoints(model_dir: Path) -> None:
    for ckpt in model_dir.glob("fold_*.ckpt"):
        try:
            ckpt.unlink()
            log.info(f"Deleted fold checkpoint: {ckpt.name}")
        except Exception as e:
            log.warning(f"Could not delete {ckpt}: {e}")


def train_all_models_and_ensemble(cfg: DictConfig):
    model_configs = {
        "model1": "configs/model/model1.yaml",
        "model2": "configs/model/model2.yaml",
        "model3": "configs/model/model3.yaml",
    }

    for model_name, config_path in model_configs.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name}")
        print(f"{'='*50}")

        model_cfg = OmegaConf.load(config_path)
        OmegaConf.update(cfg, "model", model_cfg, merge=False)
        OmegaConf.update(cfg, "model_name", model_name, merge=False)

        train_model(cfg)

        model_dir = Path(cfg.data.output_dir) / "models" / model_name
        try:
            checkpoint_path = get_best_checkpoint(model_dir)
            onnx_path = model_dir / "model.onnx"

            convert_to_onnx(str(checkpoint_path), str(onnx_path), model_cfg)
            print(f"onx exported: {onnx_path}")
            trt_path = model_dir / "model.trt"
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "src/cassava_classifier/pipelines/trt_convert.py",
                        "--onnx",
                        str(onnx_path),
                        "--output",
                        str(trt_path),
                        "--fp16",
                    ],
                    check=True,
                )
                print(f"tensorRT exported: {trt_path}")
            except Exception as e:
                print(f"tensorrt conversion failed for {model_name}: {e}")
            if not cfg.get("debug", False):
                cleanup_fold_checkpoints(model_dir)
            else:
                print("Debug mode: keeping fold checkpoints")

        except Exception as e:
            print(f"onnx conversion failed for {model_name}: {e}")

    try:
        ensemble_predict(cfg)
        print("ensemble prediction completed")
    except Exception as e:
        print(f"ensemble prediction failed: {e}")
