"""
ScanScope: Multi-Model Medical Imaging Research Prototype

Main Flask application. Routes incoming requests to the appropriate
model through the model registry.

IMPORTANT: This is an educational research demo only. It cannot diagnose,
screen for, or rule out any disease. Do not use it for health decisions.
"""
import os
import uuid
import json
from dataclasses import asdict

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory)

import config
from models.registry import registry
from models.histopathology import HistopathologyModel
from models.chest_xray import ChestXrayModel
from validation.input_validator import validate_upload


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

    # --- Register models ---
    histo_model = HistopathologyModel()
    registry.register(
        exam_type_id='histopathology_tumor_detection',
        model=histo_model,
        display_name='Histopathology — Tumor Metastasis Detection',
        modality='Histopathology (H&E-stained tissue patch)',
        task_description=(
            'Classify a histopathology image patch as Tumor metastasis '
            'or Normal tissue, using a ResNet-18 trained on Camelyon16 '
            'lymph node sections.'
        ),
        body_region='Lymph Node Tissue',
    )

    chest_model = ChestXrayModel()
    registry.register(
        exam_type_id='chest_xray_pneumonia',
        model=chest_model,
        display_name='Chest X-Ray — Pneumonia Detection',
        modality='X-Ray (anterior-posterior chest radiograph)',
        task_description=(
            'Classify a chest X-ray as NORMAL or PNEUMONIA, using a '
            'ResNet-18 fine-tuned on the Kermany pediatric chest X-ray '
            'dataset.'
        ),
        body_region='Chest / Lungs',
    )

    # --- Routes ---

    @app.route('/')
    def index():
        """Serve the main page."""
        return render_template('index.html')

    @app.route('/api/exam-types', methods=['GET'])
    def get_exam_types():
        """Return all registered examination types."""
        return jsonify({
            'exam_types': registry.list_examination_types()
        })

    @app.route('/api/predict', methods=['POST'])
    def predict():
        """
        Accept an image upload and examination type, validate, and
        route to the appropriate model.
        """
        # Check exam type
        exam_type = request.form.get('exam_type')
        if not exam_type:
            return jsonify({
                'error': 'No examination type selected.',
                'is_supported': False,
            }), 400

        # Check file
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file uploaded.',
                'is_supported': False,
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected.',
                'is_supported': False,
            }), 400

        # Save to temp location
        safe_name = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, safe_name)
        try:
            file.save(file_path)

            # Validate
            validation = validate_upload(file_path, file.filename)
            if not validation.is_valid:
                return jsonify({
                    'error': validation.error_message,
                    'is_supported': False,
                }), 400

            # Route to model
            result = registry.predict(exam_type, file_path)

            # Build response
            response = {
                'predicted_label': result.predicted_label,
                'model_scores': result.model_scores,
                'model_name': result.model_name,
                'is_supported': result.is_supported,
                'error_message': result.error_message,
                'explanation': result.explanation,
                'validation_warning': validation.warning_message,
                'image_info': validation.image_info,
                'metadata': {
                    'name': result.metadata.name,
                    'version': result.metadata.version,
                    'task': result.metadata.task,
                    'architecture': result.metadata.architecture,
                    'dataset_name': result.metadata.dataset_name,
                    'dataset_source': result.metadata.dataset_source,
                    'dataset_license': result.metadata.dataset_license,
                    'dataset_size': result.metadata.dataset_size,
                    'labels': result.metadata.labels,
                    'input_description': result.metadata.input_description,
                    'preprocessing_description': result.metadata.preprocessing_description,
                    'training_description': result.metadata.training_description,
                    'limitations': result.metadata.limitations,
                    'evaluation_metrics': result.metadata.evaluation_metrics,
                    'is_trained': result.metadata.is_trained,
                    'is_finetuned': result.metadata.is_finetuned,
                    'pretrained_source': result.metadata.pretrained_source,
                    'citation': result.metadata.citation,
                    'notes': result.metadata.notes,
                },
            }
            return jsonify(response)

        finally:
            # Clean up uploaded file — do not retain user uploads
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    @app.route('/api/sample-predict', methods=['POST'])
    def sample_predict():
        """
        Run prediction on a built-in sample image.
        Used for the demo mode so users don't need their own images.
        """
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request.'}), 400

        exam_type = data.get('exam_type')
        sample_name = data.get('sample_name')

        if not exam_type or not sample_name:
            return jsonify({'error': 'exam_type and sample_name required.'}), 400

        # Sanitize sample name to prevent path traversal
        safe_sample = os.path.basename(sample_name)
        sample_path = os.path.join(config.SAMPLES_DIR, exam_type, safe_sample)

        if not os.path.exists(sample_path):
            return jsonify({
                'error': f'Sample image not found. Demo samples may not be '
                         f'available for this examination type.',
                'is_supported': False,
            }), 404

        # Validate and predict
        validation = validate_upload(sample_path, safe_sample)
        if not validation.is_valid:
            return jsonify({
                'error': validation.error_message,
                'is_supported': False,
            }), 400

        result = registry.predict(exam_type, sample_path)

        response = {
            'predicted_label': result.predicted_label,
            'model_scores': result.model_scores,
            'model_name': result.model_name,
            'is_supported': result.is_supported,
            'error_message': result.error_message,
            'explanation': result.explanation,
            'validation_warning': validation.warning_message,
            'image_info': validation.image_info,
            'metadata': {
                'name': result.metadata.name,
                'version': result.metadata.version,
                'task': result.metadata.task,
                'architecture': result.metadata.architecture,
                'dataset_name': result.metadata.dataset_name,
                'dataset_source': result.metadata.dataset_source,
                'dataset_license': result.metadata.dataset_license,
                'dataset_size': result.metadata.dataset_size,
                'labels': result.metadata.labels,
                'input_description': result.metadata.input_description,
                'preprocessing_description': result.metadata.preprocessing_description,
                'training_description': result.metadata.training_description,
                'limitations': result.metadata.limitations,
                'evaluation_metrics': result.metadata.evaluation_metrics,
                'is_trained': result.metadata.is_trained,
                'is_finetuned': result.metadata.is_finetuned,
                'pretrained_source': result.metadata.pretrained_source,
                'citation': result.metadata.citation,
                'notes': result.metadata.notes,
            },
        }
        return jsonify(response)

    @app.route('/api/samples/<exam_type>', methods=['GET'])
    def list_samples(exam_type):
        """List available sample images for an examination type."""
        safe_type = os.path.basename(exam_type)
        samples_dir = os.path.join(config.SAMPLES_DIR, safe_type)
        if not os.path.isdir(samples_dir):
            return jsonify({'samples': []})

        samples = []
        for fname in sorted(os.listdir(samples_dir)):
            ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
            if ext in config.ALLOWED_EXTENSIONS:
                samples.append({
                    'name': fname,
                    'url': f'/static/samples/{safe_type}/{fname}',
                })
        return jsonify({'samples': samples})

    return app


if __name__ == '__main__':
    app = create_app()
    print("\n" + "=" * 60)
    print("  ScanScope: Multi-Model Medical Imaging Research Prototype")
    print("=" * 60)
    print("\n  ⚠  EDUCATIONAL RESEARCH DEMO ONLY")
    print("  This app cannot diagnose, screen for, or rule out")
    print("  any disease. Do not use it for health decisions.\n")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
