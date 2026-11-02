"""Abstract base class for all ScanScope models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelMetadata:
    """Metadata describing a model's provenance and capabilities."""
    name: str
    version: str
    task: str
    architecture: str
    dataset_name: str
    dataset_source: str
    dataset_license: str
    dataset_size: str
    labels: list
    input_description: str
    preprocessing_description: str
    training_description: str
    limitations: list
    evaluation_metrics: Optional[dict] = None
    is_trained: bool = False
    is_finetuned: bool = False
    pretrained_source: Optional[str] = None
    citation: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class PredictionResult:
    """Result from a model prediction."""
    predicted_label: str
    model_scores: dict  # label -> raw score (NOT calibrated probability)
    model_name: str
    metadata: ModelMetadata
    is_supported: bool = True
    error_message: Optional[str] = None
    explanation: Optional[str] = None
    attention_map: Optional[str] = None  # base64-encoded image if available


class BaseModel(ABC):
    """Abstract base class all ScanScope models must implement."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            PredictionResult with the model's output.
        """
        pass

    def get_unsupported_result(self, reason: str = "") -> PredictionResult:
        """Return a safe 'unsupported' result instead of running inference."""
        metadata = self.get_metadata()
        return PredictionResult(
            predicted_label="Unsupported Input",
            model_scores={},
            model_name=metadata.name,
            metadata=metadata,
            is_supported=False,
            error_message=reason or (
                "This image does not appear compatible with this model's "
                "expected input format. No prediction was made."
            ),
        )
