from pathlib import Path

import cv2
import mlflow
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from ..data.datamodule import CassavaDataModule
from ..models.model import CassavaLightningModule
from ..utils.preprocessing import clean_labels

torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([type(omegaconf.DictConfig({}))])


def load_checkpoint(model_path: str, model_config):
    model = CassavaLightningModule.load_from_checkpoint(
        model_path, model_config=model_config, map_location="cpu", weights_only=False
    )
    model.eval()
    return model


def preprocess_image(image_path: str, img_size: int):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose(
        [
            Resize(height=img_size, width=img_size),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
    )

    augmented = transform(image=image)
    return augmented["image"].unsqueeze(0)


def predict_single_image(model, image_tensor, device="cpu"):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().cpu().numpy()


def predict_with_ensemble(
    image_path: str,
    cfg: DictConfig,
    device="cuda" if torch.cuda.is_available() else "cpu",
    weights=None,
):
    print("🔄 Loading ensemble of 3 models for prediction (3rd place solution)")

    if weights is None:
        weights = [0.4, 0.3, 0.3]

    model_configs = [
        OmegaConf.load("configs/model/model1.yaml"),
        OmegaConf.load("configs/model/model2.yaml"),
        OmegaConf.load("configs/model/model3.yaml"),
    ]
    model_dirs = [
        Path(cfg.data.output_dir) / "models" / "model1",
        Path(cfg.data.output_dir) / "models" / "model2",
        Path(cfg.data.output_dir) / "models" / "model3",
    ]
    model_paths = [str(d / "model_best.ckpt") for d in model_dirs]
    models = []
    for path, model_cfg in zip(model_paths, model_configs):
        model = load_checkpoint(path, model_cfg)
        model.to(device)
        model.eval()
        models.append(model)

    predictions = []

    img_tensor = preprocess_image(image_path, model_configs[0].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = models[0](img_tensor)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())

    img_tensor = preprocess_image(image_path, model_configs[1].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        parts = [
            img_tensor[:, :, :224, :224],
            img_tensor[:, :, :224, 224:],
            img_tensor[:, :, 224:, :224],
            img_tensor[:, :, 224:, 224:],
        ]
        features_list = []
        for part in parts:
            features = models[1].backbone(part)
            features_list.append(features)

        if hasattr(models[1], "attention") and models[1].use_attention:
            features = models[1].attention(features_list)
        else:
            features = torch.stack(features_list).mean(dim=0)

        logits = models[1].head(features)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())

    img_tensor = preprocess_image(image_path, model_configs[2].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        parts = [
            img_tensor[:, :, :224, :224],
            img_tensor[:, :, :224, 224:],
            img_tensor[:, :, 224:, :224],
            img_tensor[:, :, 224:, 224:],
        ]

        features_list = []
        for part in parts:
            features = models[2].backbone(part)
            features_list.append(features)

        if hasattr(models[2], "attention") and models[2].use_attention:
            features = models[2].attention(features_list)
        else:
            features = torch.stack(features_list).mean(dim=0)

        logits = models[2].head(features)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())

    weighted_sum = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_sum += weights[i] * pred
    final_probs = weighted_sum / np.sum(weighted_sum)
    pred_class = int(np.argmax(final_probs))
    print(f"✅ Ensemble prediction completed. Class: {pred_class}, Weights: {weights}")
    return pred_class, final_probs[0]


def ensemble_predict(cfg: DictConfig):
    print(f"\n{'='*50}")
    print("RUNNING ENSEMBLE PREDICTION")
    print(f"{'='*50}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = Path(cfg.data.dataroot) / "train.csv"
    df = pd.read_csv(train_csv)
    df = clean_labels(df, cfg.data.dataroot)
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )
    _, val_idx = list(skf.split(df, df["label"]))[-1]
    val_df = df.iloc[val_idx].reset_index(drop=True)
    data_module = CassavaDataModule(
        val_df,
        val_df,
        model_config=cfg.model,
        dataroot=cfg.data.dataroot,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()
    model_dirs = [
        Path(cfg.data.output_dir) / "models" / f"model{i}" for i in range(1, 4)
    ]
    model_paths = [str(d / "model_best.ckpt") for d in model_dirs]

    for path in model_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    configs = []
    for i in range(1, 4):
        cfg_i = OmegaConf.load(f"configs/model/model{i}.yaml")
        cfg_i.pretrained = False
        configs.append(cfg_i)

    models = []
    for path, model_cfg in zip(model_paths, configs):
        model = load_checkpoint(path, model_cfg)
        model.to(device)
        model.eval()
        models.append(model)

    all_preds = []
    all_targets = []

    torch.set_grad_enabled(False)
    for batch in tqdm(val_loader, desc="Ensemble Predict"):
        x, y = batch
        x = x.to(device)
        ensemble_probs = []

        for model, model_cfg in zip(models, configs):
            if x.shape[-1] != model_cfg.img_size:
                x_resized = F.interpolate(
                    x.float(), size=(model_cfg.img_size, model_cfg.img_size)
                )
            else:
                x_resized = x.float()

            with torch.no_grad():
                probs = torch.softmax(model(x_resized), dim=1)
                ensemble_probs.append(probs)

        avg_probs = torch.stack(ensemble_probs).mean(0)
        preds = torch.argmax(avg_probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow.experiment_name)
    with mlflow.start_run(run_name="ensemble_all_models", nested=True):
        mlflow.log_metrics({"ensemble_accuracy": acc, "ensemble_f1": f1})
        mlflow.log_params(
            {
                "ensembled_models": ["model1", "model2", "model3"],
                "debug_mode": cfg.get("debug", False),
            }
        )

    print("\nENSEMBLE RESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*50}")
