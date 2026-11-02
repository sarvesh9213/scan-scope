"""
Model registry and router for ScanScope.

Each examination type maps to exactly one model. The router dispatches
only to models explicitly registered and verified as ready.
"""
from typing import Dict, Optional
from models.base import BaseModel, PredictionResult


class ModelRegistry:
    """Registry that maps examination types to their models."""

    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._exam_descriptions: Dict[str, dict] = {}

    def register(self, exam_type_id: str, model: BaseModel,
                 display_name: str, modality: str, task_description: str,
                 body_region: str):
        """
        Register a model for a specific examination type.

        Args:
            exam_type_id: Unique identifier (e.g., 'chest_xray_pneumonia').
            model: An instance of BaseModel.
            display_name: Human-readable name shown in the UI.
            modality: Imaging modality (e.g., 'X-Ray', 'Histopathology').
            task_description: What the model predicts.
            body_region: Body region this applies to.
        """
        self._models[exam_type_id] = model
        self._exam_descriptions[exam_type_id] = {
            'id': exam_type_id,
            'display_name': display_name,
            'modality': modality,
            'task_description': task_description,
            'body_region': body_region,
            'is_ready': model.is_ready(),
            'is_trained': model.get_metadata().is_trained,
        }

    def get_model(self, exam_type_id: str) -> Optional[BaseModel]:
        """Get the model for an examination type, or None."""
        return self._models.get(exam_type_id)

    def predict(self, exam_type_id: str, image_path: str) -> PredictionResult:
        """
        Route an image to the correct model for the given examination type.

        Returns an unsupported result if the exam type is unknown or the
        model is not ready.
        """
        model = self.get_model(exam_type_id)
        if model is None:
            # Create a minimal error result — no model to delegate to
            from models.base import ModelMetadata
            return PredictionResult(
                predicted_label="Unknown Examination Type",
                model_scores={},
                model_name="None",
                metadata=ModelMetadata(
                    name="None", version="N/A", task="N/A",
                    architecture="N/A", dataset_name="N/A",
                    dataset_source="N/A", dataset_license="N/A",
                    dataset_size="N/A", labels=[], input_description="N/A",
                    preprocessing_description="N/A",
                    training_description="N/A", limitations=[],
                ),
                is_supported=False,
                error_message=(
                    f"Examination type '{exam_type_id}' is not registered. "
                    "No model was invoked."
                ),
            )

        if not model.is_ready():
            return model.get_unsupported_result(
                "This model is registered but not yet ready (weights may "
                "not be loaded). No prediction was made."
            )

        return model.predict(image_path)

    def list_examination_types(self) -> list:
        """Return all registered examination types with their descriptions."""
        return list(self._exam_descriptions.values())


# Global singleton registry
registry = ModelRegistry()
