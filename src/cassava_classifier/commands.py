# # working/Cassava-Disease-Detection/src/cassava_classifier/commands.py
# import os
# import sys

# import hydra
# import torch
# from omegaconf import DictConfig

# from cassava_classifier.pipelines.train import (
#     train_all_models_and_ensemble,
#     train_model,
# )

# # Add src to Python path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# # Absolute imports

# print("CUDA available:", torch.cuda.is_available())
# print("GPU count:", torch.cuda.device_count())


# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
# def main(cfg: DictConfig):
#     if cfg.get("run_full", False):
#         train_all_models_and_ensemble(cfg)
#     elif cfg.get("predict", False):
#         from cassava_classifier.pipelines.infer import (
#             load_checkpoint,
#             predict_single_image,
#             preprocess_image,
#         )

#         # Load model and predict
#         model_config = cfg.model
#         model = load_checkpoint(cfg.predict.model_path, model_config)
#         image_tensor = preprocess_image(cfg.predict.image_path, model_config.img_size)
#         pred_class, probs = predict_single_image(model, image_tensor)
#         print(f"Predicted class: {pred_class}")
#         print(f"Probabilities: {probs}")
#     else:
#         train_model(cfg)


# if __name__ == "__main__":
#     main()

# working/Cassava-Disease-Detection/src/cassava_classifier/commands.py
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cassava_classifier.pipelines.train import (
    train_all_models_and_ensemble,
    train_model,
)

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("run_full", False):
        train_all_models_and_ensemble(cfg)
    elif cfg.get("predict", False):
        from cassava_classifier.pipelines.infer import (
            load_checkpoint,
            predict_single_image,
            predict_with_ensemble,  # Added ensemble prediction function
            preprocess_image,
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Check if we should use ensemble prediction
        use_ensemble = cfg.predict.get("use_ensemble", False)
        
        if use_ensemble:
            print("🔮 Running ENSEMBLE prediction with all 3 models (3rd place solution)")
            # Get ensemble weights from config or use defaults [0.4, 0.3, 0.3]
            ensemble_weights = cfg.predict.get("ensemble_weights", [0.4, 0.3, 0.3])
            print(f"Using ensemble weights: {ensemble_weights}")
            
            # Run ensemble prediction
            pred_class, probs = predict_with_ensemble(
                cfg.predict.image_path,
                cfg,
                device=device,
                weights=ensemble_weights
            )
        else:
            print(f"🔍 Running single model prediction with: {cfg.predict.model_path}")
            # Load model and predict (existing single-model flow)
            model_config = cfg.model
            model = load_checkpoint(cfg.predict.model_path, model_config)
            model.to(device)
            image_tensor = preprocess_image(cfg.predict.image_path, model_config.img_size)
            pred_class, probs = predict_single_image(model, image_tensor, device=device)
        
        # Print results
        class_names = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]  # Disease class names
        print(f"\n{'='*50}")
        print(f"RESULT: Predicted class {pred_class} - {class_names[pred_class]}")
        print(f"{'='*50}")
        for i, prob in enumerate(probs):
            print(f"{class_names[i]}: {prob:.4f}")
    else:
        train_model(cfg)


if __name__ == "__main__":
    main()