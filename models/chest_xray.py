"""
Chest X-ray pneumonia classification model for ScanScope.

Dataset: Kermany et al. "Chest X-Ray Images (Pneumonia)"
  - Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
  - Also available on Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - License: CC BY 4.0
  - 5,863 JPEG images in two categories: PNEUMONIA, NORMAL
  - Anterior-posterior chest X-rays from pediatric patients (ages 1–5)
    at Guangzhou Women and Children's Medical Center
  - Citation: Kermany, D.S. et al. "Identifying Medical Diagnoses and
    Treatable Diseases by Image-Based Deep Learning." Cell, 172(5),
    1122–1131.e9, 2018.

Model: ResNet-18 pretrained on ImageNet, fine-tuned on this dataset.
Status: Training may or may not complete depending on environment.
"""
import os
import traceback
from typing import Optional

from PIL import Image

from models.base import BaseModel, ModelMetadata, PredictionResult
import config


class ChestXrayModel(BaseModel):
    """
    Classifies chest X-rays as NORMAL vs PNEUMONIA.

    Uses a ResNet-18 fine-tuned on the Kermany chest X-ray dataset.
    Input: Chest X-ray image (anterior-posterior view).
    Output: Binary classification — 'NORMAL' or 'PNEUMONIA' (dataset labels).
    """

    WEIGHTS_FILENAME = "chest_xray_resnet18.pth"

    def __init__(self):
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._eval_metrics: Optional[dict] = None

        # Check if we have saved evaluation metrics
        metrics_path = os.path.join(config.WEIGHTS_DIR, "chest_xray_metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                self._eval_metrics = json.load(f)

        self._metadata = ModelMetadata(
            name="Chest X-Ray Pneumonia Classifier (ResNet-18)",
            version="1.0",
            task="Binary classification of chest X-ray images: "
                 "NORMAL vs PNEUMONIA",
            architecture="ResNet-18 (torchvision, pretrained on ImageNet, "
                         "final FC layer replaced for 2-class output)",
            dataset_name="Chest X-Ray Images (Pneumonia) — Kermany et al.",
            dataset_source="https://data.mendeley.com/datasets/rscbjbr9sj/2 "
                           "(also on Kaggle)",
            dataset_license="CC BY 4.0",
            dataset_size="5,863 JPEG images (train: 5,216; test: 624; "
                         "val: 16). Note: significant class imbalance — "
                         "~73.5% PNEUMONIA in training set. Pediatric "
                         "patients ages 1–5.",
            labels=["NORMAL", "PNEUMONIA"],
            input_description="Anterior-posterior chest X-ray image (JPEG). "
                              "Originally from pediatric patients. The model "
                              "resizes input to 224×224.",
            preprocessing_description="Image is resized to 224×224 pixels, "
                                      "converted to RGB (X-rays are often "
                                      "grayscale — converted to 3-channel), "
                                      "normalized with ImageNet mean "
                                      "[0.485, 0.456, 0.406] and std "
                                      "[0.229, 0.224, 0.225].",
            training_description="NOT YET TRAINED. When trained: ResNet-18 "
                                 "pretrained on ImageNet, fine-tuned with "
                                 "cross-entropy loss, Adam optimizer, "
                                 "lr=1e-4, for up to 10 epochs with early "
                                 "stopping. Data augmentation: random "
                                 "horizontal flip, random rotation ±10°, "
                                 "random affine translation.",
            limitations=[
                "Designed for anterior-posterior chest X-rays only.",
                "Trained on pediatric patients (ages 1–5); performance on "
                "adult X-rays is unknown and likely different.",
                "Binary classification only (NORMAL vs PNEUMONIA); does NOT "
                "distinguish bacterial vs. viral pneumonia, or other lung "
                "conditions.",
                "Significant class imbalance in training data (~73.5% PNEUMONIA).",
                "Model scores are NOT calibrated probabilities of disease.",
                "This is a research tool. Results must not be used for diagnosis.",
                "A normal-looking result does NOT rule out pneumonia or any "
                "other condition.",
            ],
            is_trained=False,
            is_finetuned=False,
            pretrained_source="torchvision ResNet-18 pretrained on ImageNet "
                              "(https://pytorch.org/vision/stable/models.html)",
            citation="Kermany, D.S. et al. 'Identifying Medical Diagnoses and "
                     "Treatable Diseases by Image-Based Deep Learning.' Cell, "
                     "172(5), 1122–1131.e9, 2018.",
            evaluation_metrics=self._eval_metrics,
            notes=None,
        )
        self._try_load()

    def _try_load(self):
        """Attempt to load trained model weights."""
        try:
            import torch
            import torchvision.models as models

            weights_path = os.path.join(
                config.WEIGHTS_DIR, self.WEIGHTS_FILENAME
            )
            if not os.path.exists(weights_path):
                self._load_error = (
                    "Model weights not found. Training has not been completed. "
                    "This model is registered but cannot make predictions until "
                    "training is run via training/train_chest_xray.py."
                )
                self._metadata.notes = self._load_error
                return

            # Build model architecture
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 2)

            # Load saved weights
            state_dict = torch.load(weights_path, map_location='cpu',
                                    weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self._ready = True
            self._metadata.is_trained = True
            self._metadata.is_finetuned = True
            self._metadata.training_description = (
                "ResNet-18 pretrained on ImageNet, fine-tuned on the Kermany "
                "chest X-ray dataset with cross-entropy loss, Adam optimizer "
                "(lr=1e-4), for up to 10 epochs with early stopping."
            )

            # Reload metrics if available
            metrics_path = os.path.join(
                config.WEIGHTS_DIR, "chest_xray_metrics.json"
            )
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    self._eval_metrics = json.load(f)
                    self._metadata.evaluation_metrics = self._eval_metrics

        except Exception as e:
            self._load_error = f"Failed to load model: {e}\n{traceback.format_exc()}"

    def get_metadata(self) -> ModelMetadata:
        return self._metadata

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, image_path: str) -> PredictionResult:
        if not self._ready:
            return self.get_unsupported_result(
                self._load_error or "Model is not loaded or trained."
            )

        try:
            import torch
            import torchvision.transforms as transforms

            img = Image.open(image_path).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

            with torch.no_grad():
                outputs = self._model(tensor)
                # outputs shape: [1, 2] — logits for [NORMAL, PNEUMONIA]
                scores = torch.softmax(outputs, dim=1).squeeze()
                predicted_idx = torch.argmax(scores).item()

            labels = ["NORMAL", "PNEUMONIA"]
            predicted_label = labels[predicted_idx]

            return PredictionResult(
                predicted_label=predicted_label,
                model_scores={
                    "NORMAL": round(scores[0].item(), 4),
                    "PNEUMONIA": round(scores[1].item(), 4),
                },
                model_name=self._metadata.name,
                metadata=self._metadata,
                is_supported=True,
                explanation=(
                    f"The model assigned scores of {scores[0].item():.4f} for "
                    f"'NORMAL' and {scores[1].item():.4f} for 'PNEUMONIA'. "
                    f"These are softmax-transformed model scores, NOT "
                    f"calibrated probabilities. The model has not undergone "
                    f"probability calibration, so these scores should not be "
                    f"interpreted as the chance that the patient has pneumonia."
                ),
            )

        except Exception as e:
            return self.get_unsupported_result(
                f"Error during inference: {str(e)}"
            )
