"""Application configuration for ScanScope."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload settings
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Model weights directory
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

# NCRF checkpoint path (sibling project)
NCRF_BASE_CKPT = os.path.join(
    BASE_DIR, '..', 'NCRF', 'ckpt', 'resnet18_base.ckpt'
)

# Sample images directory
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
