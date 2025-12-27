# # src/cassava_classifier/pipelines/infer.py
# from pathlib import Path

# import cv2
# import pandas as pd
# import omegaconf
# import torch
# import torch.nn.functional as F
# from albumentations import Compose, Normalize, Resize
# from albumentations.pytorch import ToTensorV2
# from omegaconf import DictConfig, OmegaConf
# from sklearn.metrics import accuracy_score, f1_score

# from ..data.datamodule import CassavaDataModule
# from ..models.model import CassavaLightningModule
# from ..utils.preprocessing import clean_labels
# import mlflow
# from tqdm import tqdm

# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([type(omegaconf.DictConfig({}))])

# def load_checkpoint(model_path: str, model_config):
#     """Load trained model from checkpoint"""
#     model = CassavaLightningModule.load_from_checkpoint(
#         model_path,
#         model_config=model_config,
#         map_location="cpu",
#         weights_only=False  # ⚠️ Required to load non-weight data (e.g., OmegaConf)
#     )
#     model.eval()
#     return model


# def preprocess_image(image_path: str, img_size: int):
#     """Preprocess single image for inference"""
#     image = cv2.imread(str(image_path))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     transform = Compose(
#         [
#             Resize(height=img_size, width=img_size),
#             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ToTensorV2(),
#         ]
#     )

#     augmented = transform(image=image)
#     return augmented["image"].unsqueeze(0)  # Add batch dimension


# def predict_single_image(model, image_tensor, device="cpu"):
#     """Run prediction on single image"""
#     image_tensor = image_tensor.to(device)
#     with torch.no_grad():
#         logits = model(image_tensor)
#         probabilities = F.softmax(logits, dim=1)
#         predicted_class = torch.argmax(probabilities, dim=1)
#     return predicted_class.item(), probabilities.squeeze().cpu().numpy()

# # def ensemble_predict(cfg: DictConfig):
# #     """
# #     Ensemble is used for evaluation only.
# #     It is NOT used for production inference or deployment.
# #     """
# #     print(f"\n{'='*50}")
# #     print("RUNNING ENSEMBLE PREDICTION")
# #     print(f"{'='*50}")

# #     # Load validation data
# #     train_csv = Path(cfg.data.dataroot) / "train.csv"
# #     df = pd.read_csv(train_csv)
# #     df = clean_labels(df, cfg.data.dataroot)

# #     # Use last fold's val split (for consistency)
# #     from sklearn.model_selection import StratifiedKFold

# #     skf = StratifiedKFold(
# #         n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
# #     )
# #     _, val_idx = list(skf.split(df, df["label"]))[-1]
# #     val_df = df.iloc[val_idx].reset_index(drop=True)

# #     # Create dataloader
# #     data_module = CassavaDataModule(
# #         val_df,
# #         val_df,
# #         model_config=cfg.model,
# #         dataroot=cfg.data.dataroot,
# #         batch_size=cfg.train.batch_size,
# #         num_workers=cfg.train.num_workers,
# #     )
# #     data_module.setup()
# #     val_loader = data_module.val_dataloader()

# #     # Load all 3 models - FIXED PATHS

    
# #     model_dirs = [
# #         Path(cfg.data.output_dir) / "models" / "model1",
# #         Path(cfg.data.output_dir) / "models" / "model2",
# #         Path(cfg.data.output_dir) / "models" / "model3",
# #     ]
    
# #     model_paths = []
# #     for model_dir in model_dirs:
# #         checkpoint = model_dir / "model_best.ckpt"
# #         if not checkpoint.exists():
# #             raise FileNotFoundError(f"Missing model_best.ckpt in {model_dir}")
# #         model_paths.append(str(checkpoint))

# #     configs = [
# #         OmegaConf.load("configs/model/model1.yaml"),
# #         OmegaConf.load("configs/model/model2.yaml"),
# #         OmegaConf.load("configs/model/model3.yaml"),
# #     ]
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     models = []
# #     for path, model_cfg in zip(model_paths, configs):
# #         model = load_checkpoint(path, model_cfg)  # ← Use fixed function
# #         model.to(device)
# #         model.eval()
# #         models.append(model)
        
# #     # Run ensemble
# #     all_preds = []
# #     all_targets = []
# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     for batch in val_loader:
# #         x, y = batch
# #         x = x.to(device)
# #         ensemble_probs = []

# #         for model, model_cfg in zip(models, configs):
# #             if x.shape[-1] != model_cfg.img_size:
# #                 x_resized = F.interpolate(
# #                     x.float(), size=(model_cfg.img_size, model_cfg.img_size)
# #                 )
# #             else:
# #                 x_resized = x.float()

# #             with torch.no_grad():
# #                 probs = torch.softmax(model(x_resized), dim=1)
# #                 ensemble_probs.append(probs)

# #         avg_probs = torch.stack(ensemble_probs).mean(0)
# #         preds = torch.argmax(avg_probs, dim=1)
# #         all_preds.extend(preds.cpu().numpy())
# #         all_targets.extend(y.numpy())

# #     # Metrics
# #     acc = accuracy_score(all_targets, all_preds)
# #     f1 = f1_score(all_targets, all_preds, average="weighted")
    
# #     mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
# #     mlflow.set_experiment(cfg.train.mlflow.experiment_name)
    
# #     with mlflow.start_run(run_name="ensemble_all_models"):
# #         mlflow.log_metrics({
# #             "ensemble_accuracy": acc,
# #             "ensemble_f1": f1
# #         })
# #         mlflow.log_params({
# #             "ensembled_models": ["model1", "model2", "model3"],
# #             "debug_mode": cfg.get("debug", False)
# #         })
    
# #     print("\nENSEMBLE RESULTS:")
# #     print(f"Accuracy: {acc:.4f}")
# #     print(f"F1-Score: {f1:.4f}")
# #     print(f"{'='*50}")
# #     print(f"\n{'='*50}")
    
# def ensemble_predict(cfg: DictConfig):
#     """
#     Ensemble prediction for evaluation. Not for production deployment.
#     """
#     print(f"\n{'='*50}")
#     print("RUNNING ENSEMBLE PREDICTION")
#     print(f"{'='*50}")

#     # Determine device first
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load validation data
#     train_csv = Path(cfg.data.dataroot) / "train.csv"
#     df = pd.read_csv(train_csv)
#     df = clean_labels(df, cfg.data.dataroot)

#     # Use last fold's val split
#     from sklearn.model_selection import StratifiedKFold
#     skf = StratifiedKFold(
#         n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
#     )
#     _, val_idx = list(skf.split(df, df["label"]))[-1]
#     val_df = df.iloc[val_idx].reset_index(drop=True)

#     # Create dataloader
#     data_module = CassavaDataModule(
#         val_df,
#         val_df,
#         model_config=cfg.model,
#         dataroot=cfg.data.dataroot,
#         batch_size=cfg.train.batch_size,
#         num_workers=cfg.train.num_workers,
#     )
#     data_module.setup()
#     val_loader = data_module.val_dataloader()

#     # Load all 3 models
#     model_dirs = [
#         Path(cfg.data.output_dir) / "models" / f"model{i}" for i in range(1, 4)
#     ]
#     model_paths = [str(d / "model_best.ckpt") for d in model_dirs]

#     for path in model_paths:
#         if not Path(path).exists():
#             raise FileNotFoundError(f"Missing checkpoint: {path}")

#     configs = []
#     for i in range(1, 4):
#         cfg_i = OmegaConf.load(f"configs/model/model{i}.yaml")
#         cfg_i.pretrained = False 
#         configs.append(cfg_i)

#     models = []
#     for path, model_cfg in zip(model_paths, configs):
#         model = load_checkpoint(path, model_cfg)
#         model.to(device)
#         model.eval()
#         models.append(model)

#     # Run ensemble
#     all_preds = []
#     all_targets = []

#     torch.set_grad_enabled(False)
#     for batch in tqdm(val_loader,desc ='Ensemble Predict'):
#         x, y = batch
#         x = x.to(device)
#         ensemble_probs = []

#         for model, model_cfg in zip(models, configs):
#             # Resize to model's expected input if needed
#             if x.shape[-1] != model_cfg.img_size:
#                 x_resized = F.interpolate(x.float(), size=(model_cfg.img_size, model_cfg.img_size))
#             else:
#                 x_resized = x.float()

#             with torch.no_grad():
#                 probs = torch.softmax(model(x_resized), dim=1)
#                 ensemble_probs.append(probs)

#         # Average predictions
#         avg_probs = torch.stack(ensemble_probs).mean(0)
#         preds = torch.argmax(avg_probs, dim=1)

#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(y.numpy())

#     # Metrics
#     acc = accuracy_score(all_targets, all_preds)
#     f1 = f1_score(all_targets, all_preds, average="weighted")

#     # Log metrics to MLflow
#     mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
#     mlflow.set_experiment(cfg.train.mlflow.experiment_name)
#     with mlflow.start_run(run_name="ensemble_all_models", nested=True):
#         mlflow.log_metrics({"ensemble_accuracy": acc, "ensemble_f1": f1})
#         mlflow.log_params({"ensembled_models": ["model1", "model2", "model3"], "debug_mode": cfg.get("debug", False)})

#     print("\nENSEMBLE RESULTS:")
#     print(f"Accuracy: {acc:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"{'='*50}")

# src/cassava_classifier/pipelines/infer.py
from pathlib import Path
import numpy as np  # Added for ensemble prediction
import cv2
import pandas as pd
import omegaconf
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from ..data.datamodule import CassavaDataModule
from ..models.model import CassavaLightningModule
from ..utils.preprocessing import clean_labels
import mlflow
from tqdm import tqdm

torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([type(omegaconf.DictConfig({}))])

def load_checkpoint(model_path: str, model_config):
    """Load trained model from checkpoint"""
    model = CassavaLightningModule.load_from_checkpoint(
        model_path,
        model_config=model_config,
        map_location="cpu",
        weights_only=False  # ⚠️ Required to load non-weight data (e.g., OmegaConf)
    )
    model.eval()
    return model


def preprocess_image(image_path: str, img_size: int):
    """Preprocess single image for inference"""
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
    return augmented["image"].unsqueeze(0)  # Add batch dimension


def predict_single_image(model, image_tensor, device="cpu"):
    """Run prediction on single image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities.squeeze().cpu().numpy()


def predict_with_ensemble(image_path: str, cfg: DictConfig, device="cuda" if torch.cuda.is_available() else "cpu", weights=None):
    """
    Run inference using the ensemble of all three models as in 3rd place solution
    """
    print("🔄 Loading ensemble of 3 models for prediction (3rd place solution)")
    
    # Default weights if none provided
    if weights is None:
        weights = [0.4, 0.3, 0.3]  # vit384, vit224-A, vit224-B
    
    # Load model configurations
    model_configs = [
        OmegaConf.load("configs/model/model1.yaml"),  # vit_base_patch16_384
        OmegaConf.load("configs/model/model2.yaml"),  # vit_base_patch16_224 - A
        OmegaConf.load("configs/model/model3.yaml")   # vit_base_patch16_224 - B
    ]
    
    # Get model paths
    model_dirs = [
        Path(cfg.data.output_dir) / "models" / "model1",
        Path(cfg.data.output_dir) / "models" / "model2",
        Path(cfg.data.output_dir) / "models" / "model3",
    ]
    model_paths = [str(d / "model_best.ckpt") for d in model_dirs]
    
    # Load models
    models = []
    for path, model_cfg in zip(model_paths, model_configs):
        model = load_checkpoint(path, model_cfg)
        model.to(device)
        model.eval()
        models.append(model)
    
    # Get predictions from each model
    predictions = []
    
    # Model 1: vit_base_patch16_384 (384x384)
    img_tensor = preprocess_image(image_path, model_configs[0].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = models[0](img_tensor)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())
    
    # Model 2: vit_base_patch16_224 - A (448x448 with Pattern A)
    img_tensor = preprocess_image(image_path, model_configs[1].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        # Divide 448x448 into four 224x224 parts
        parts = [
            img_tensor[:, :, :224, :224],
            img_tensor[:, :, :224, 224:],
            img_tensor[:, :, 224:, :224],
            img_tensor[:, :, 224:, 224:]
        ]
        
        # Process each part through the model
        features_list = []
        for part in parts:
            features = models[1].backbone(part)
            features_list.append(features)
        
        # Apply attention weighting (Pattern A)
        if hasattr(models[1], 'attention') and models[1].use_attention:
            features = models[1].attention(features_list)
        else:
            features = torch.stack(features_list).mean(dim=0)
        
        # Get final prediction
        logits = models[1].head(features)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())
    
    # Model 3: vit_base_patch16_224 - B (448x448 with Pattern B)
    img_tensor = preprocess_image(image_path, model_configs[2].img_size)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        # Divide 448x448 into four 224x224 parts
        parts = [
            img_tensor[:, :, :224, :224],
            img_tensor[:, :, :224, 224:],
            img_tensor[:, :, 224:, :224],
            img_tensor[:, :, 224:, 224:]
        ]
        
        # Process each part through the model
        features_list = []
        for part in parts:
            features = models[2].backbone(part)
            features_list.append(features)
        
        # Apply attention weighting (Pattern B)
        if hasattr(models[2], 'attention') and models[2].use_attention:
            features = models[2].attention(features_list)
        else:
            features = torch.stack(features_list).mean(dim=0)
        
        # Get final prediction
        logits = models[2].head(features)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())
    
    # Weighted averaging of predictions
    weighted_sum = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_sum += weights[i] * pred
    
    # Normalize to get final probabilities
    final_probs = weighted_sum / np.sum(weighted_sum)
    pred_class = int(np.argmax(final_probs))
    
    print(f"✅ Ensemble prediction completed. Class: {pred_class}, Weights: {weights}")
    return pred_class, final_probs[0]


def ensemble_predict(cfg: DictConfig):
    """
    Ensemble prediction for evaluation. Not for production deployment.
    """
    print(f"\n{'='*50}")
    print("RUNNING ENSEMBLE PREDICTION")
    print(f"{'='*50}")

    # Determine device first
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load validation data
    train_csv = Path(cfg.data.dataroot) / "train.csv"
    df = pd.read_csv(train_csv)
    df = clean_labels(df, cfg.data.dataroot)

    # Use last fold's val split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )
    _, val_idx = list(skf.split(df, df["label"]))[-1]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Create dataloader
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

    # Load all 3 models
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

    # Run ensemble
    all_preds = []
    all_targets = []

    torch.set_grad_enabled(False)
    for batch in tqdm(val_loader,desc ='Ensemble Predict'):
        x, y = batch
        x = x.to(device)
        ensemble_probs = []

        for model, model_cfg in zip(models, configs):
            # Resize to model's expected input if needed
            if x.shape[-1] != model_cfg.img_size:
                x_resized = F.interpolate(x.float(), size=(model_cfg.img_size, model_cfg.img_size))
            else:
                x_resized = x.float()

            with torch.no_grad():
                probs = torch.softmax(model(x_resized), dim=1)
                ensemble_probs.append(probs)

        # Average predictions
        avg_probs = torch.stack(ensemble_probs).mean(0)
        preds = torch.argmax(avg_probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")

    # Log metrics to MLflow
    mlflow.set_tracking_uri(cfg.train.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow.experiment_name)
    with mlflow.start_run(run_name="ensemble_all_models", nested=True):
        mlflow.log_metrics({"ensemble_accuracy": acc, "ensemble_f1": f1})
        mlflow.log_params({"ensembled_models": ["model1", "model2", "model3"], "debug_mode": cfg.get("debug", False)})

    print("\nENSEMBLE RESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*50}")