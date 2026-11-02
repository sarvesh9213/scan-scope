"""
Histopathology patch classification model for ScanScope.

Uses the pretrained ResNet-18 (baseline, no CRF) checkpoint from the
sibling NCRF project trained on the Camelyon16 dataset.

The NCRF model architecture expects 5D input:
  [batch_size, grid_size, 3, crop_size, crop_size]
For single-patch inference, we use grid_size=1.

Dataset: Camelyon16 Grand Challenge
  - 400 whole-slide images of sentinel lymph node sections
  - Binary task: tumor metastasis vs. normal tissue
  - Patches extracted at 256×256 (resized to 224×224 for ResNet)
  - Source: https://camelyon16.grand-challenge.org/
  - The data was collected from Radboud UMC and University Medical Center Utrecht

Pretrained weights source: NCRF project by Baidu Research
  - Paper: Li & Ping, "Cancer Metastasis Detection With Neural Conditional
    Random Field", MIDL 2018
  - Repository: https://github.com/baidu-research/NCRF
  - License: Not explicitly stated in the NCRF repo; weights provided for
    research use alongside the paper
"""
import os
import sys
import traceback
from typing import Optional

import numpy as np
from PIL import Image

from models.base import BaseModel, ModelMetadata, PredictionResult
import config


# We need to import the NCRF model definition.
# Add NCRF project root to sys.path so we can import wsi.model
NCRF_ROOT = os.path.normpath(os.path.join(config.BASE_DIR, '..', 'NCRF'))


def _build_ncrf_model():
    """Build the NCRF ResNet-18 base model (no CRF) for single-patch inference."""
    import torch

    # Temporarily add NCRF to path to import the model definition
    if NCRF_ROOT not in sys.path:
        sys.path.insert(0, NCRF_ROOT)

    from wsi.model.resnet import resnet18

    # Build model: num_nodes=1 (single patch), use_crf=False (baseline)
    model = resnet18(num_classes=1, num_nodes=1, use_crf=False)
    return model


def _load_checkpoint(model, ckpt_path: str):
    """Load weights from the NCRF checkpoint into the model."""
    import torch

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # The NCRF checkpoints may store state_dict under various keys
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Filter out CRF-related keys if present (we use base model without CRF)
    filtered = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if saved from DataParallel
        clean_key = k.replace('module.', '')
        if 'crf' not in clean_key.lower():
            filtered[clean_key] = v

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


class HistopathologyModel(BaseModel):
    """
    Classifies histopathology image patches as Tumor vs. Normal.

    Uses the NCRF ResNet-18 baseline model pretrained on Camelyon16.
    Input: RGB image patch (will be resized to 224×224).
    Output: Binary classification — 'Tumor' or 'Normal' (training labels).
    """

    def __init__(self):
        self._model = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._metadata = ModelMetadata(
            name="Histopathology Patch Classifier (ResNet-18 Baseline)",
            version="1.0",
            task="Binary classification of histopathology image patches: "
                 "Tumor metastasis vs. Normal tissue",
            architecture="ResNet-18 (baseline, without CRF)",
            dataset_name="Camelyon16",
            dataset_source="https://camelyon16.grand-challenge.org/",
            dataset_license="Research use; data access requires registration "
                            "with Camelyon16 organizers. Also available at "
                            "GigaDB (http://gigadb.org/dataset/100439).",
            dataset_size="400 whole-slide images (~700 GB); training patches "
                         "sampled at 768×768 from WSIs, 200k tumor + 200k "
                         "normal for training, 20k each for validation",
            labels=["Tumor", "Normal"],
            input_description="RGB histopathology image patch, ideally "
                              "256×256 pixels extracted from H&E-stained "
                              "sentinel lymph node tissue sections. The model "
                              "resizes input to 224×224.",
            preprocessing_description="Image is resized to 224×224 pixels, "
                                      "converted to RGB, normalized with "
                                      "ImageNet mean [0.485, 0.456, 0.406] "
                                      "and std [0.229, 0.224, 0.225].",
            training_description="Pretrained by Baidu Research as part of the "
                                 "NCRF project. The ResNet-18 baseline was "
                                 "trained on patches extracted from Camelyon16 "
                                 "WSIs using binary cross-entropy loss with "
                                 "SGD (lr=0.001, momentum=0.9) for 20 epochs. "
                                 "This project loads the published checkpoint "
                                 "without additional fine-tuning.",
            limitations=[
                "Designed for H&E-stained lymph node histopathology patches only.",
                "Not validated on other tissue types, staining protocols, or scanners.",
                "The checkpoint was trained on Camelyon16 data; performance on "
                "other histopathology datasets is unknown.",
                "Single-patch inference loses the spatial context that the full "
                "NCRF (CRF) model captures across patch grids.",
                "Model scores are NOT calibrated probabilities of disease.",
                "This is a research tool. Results must not be used for diagnosis.",
            ],
            is_trained=True,
            is_finetuned=False,
            pretrained_source="NCRF project by Baidu Research "
                              "(https://github.com/baidu-research/NCRF), "
                              "checkpoint: resnet18_base.ckpt",
            citation="Li, Yi and Ping, Wei. 'Cancer Metastasis Detection With "
                     "Neural Conditional Random Field.' Medical Imaging with "
                     "Deep Learning (MIDL), 2018.",
            evaluation_metrics=None,  # We haven't run our own evaluation
            notes="Loaded from published pretrained checkpoint. No independent "
                  "evaluation on a held-out test set has been performed by this "
                  "project — the evaluation metrics from the original paper are "
                  "not reproduced here to avoid misrepresentation.",
        )
        self._try_load()

    def _try_load(self):
        """Attempt to load the model weights."""
        try:
            import torch
            ckpt_path = os.path.normpath(config.NCRF_BASE_CKPT)
            if not os.path.exists(ckpt_path):
                self._load_error = (
                    f"Checkpoint not found at {ckpt_path}. "
                    "The NCRF sibling project checkpoint is required."
                )
                return

            self._model = _build_ncrf_model()
            self._model = _load_checkpoint(self._model, ckpt_path)
            self._ready = True
        except Exception as e:
            self._load_error = f"Failed to load model: {e}\n{traceback.format_exc()}"

    def get_metadata(self) -> ModelMetadata:
        return self._metadata

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, image_path: str) -> PredictionResult:
        if not self._ready:
            return self.get_unsupported_result(
                self._load_error or "Model is not loaded."
            )

        try:
            import torch
            import torchvision.transforms as transforms

            # Load and preprocess
            img = Image.open(image_path).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            tensor = transform(img)
            # NCRF expects 5D: [batch, grid_size, channels, H, W]
            # For single patch: grid_size=1
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 224, 224]

            with torch.no_grad():
                logit = self._model(tensor)
                # logit is a scalar (after squeeze) for single patch
                logit_val = logit.item()
                # Sigmoid to get score (NOT a calibrated probability)
                score = torch.sigmoid(logit).item()

            # The NCRF model: logit > 0 → Tumor, logit < 0 → Normal
            if logit_val > 0:
                predicted_label = "Tumor"
            else:
                predicted_label = "Normal"

            return PredictionResult(
                predicted_label=predicted_label,
                model_scores={
                    "Tumor": round(score, 4),
                    "Normal": round(1 - score, 4),
                },
                model_name=self._metadata.name,
                metadata=self._metadata,
                is_supported=True,
                explanation=(
                    f"The model produced a raw logit of {logit_val:.4f}. "
                    f"After sigmoid transformation, the model score toward "
                    f"'Tumor' is {score:.4f}. This score is NOT a calibrated "
                    f"probability of disease — it is a raw model output that "
                    f"reflects the model's internal confidence, which may not "
                    f"correspond to real-world frequency of the condition."
                ),
            )

        except Exception as e:
            return self.get_unsupported_result(
                f"Error during inference: {str(e)}"
            )
