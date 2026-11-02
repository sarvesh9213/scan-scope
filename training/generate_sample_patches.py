"""
Generate sample histopathology patches for the demo mode.

This script creates synthetic colored patches that resemble
histopathology images for UI demonstration purposes ONLY.
These are NOT real tissue images and cannot be used for any
medical or research purpose beyond testing the application UI.

For real predictions, use actual histopathology image patches
from the Camelyon16 dataset.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'static', 'samples')


def create_synthetic_histopathology_patch(size=256, seed=None):
    """
    Create a synthetic image with colors reminiscent of H&E staining.
    This is purely for UI demo — NOT a real tissue image.
    """
    if seed is not None:
        np.random.seed(seed)

    # H&E staining typically has pink/purple tones
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: light pink (eosin-like)
    img[:, :, 0] = np.random.randint(200, 240, (size, size))  # R
    img[:, :, 1] = np.random.randint(170, 210, (size, size))  # G
    img[:, :, 2] = np.random.randint(190, 220, (size, size))  # B

    # Add some purple/blue spots (nuclei-like)
    n_spots = np.random.randint(20, 60)
    for _ in range(n_spots):
        cx, cy = np.random.randint(10, size - 10, 2)
        r = np.random.randint(3, 12)
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x*x + y*y <= r*r
        y_coords = np.clip(cy + np.arange(-r, r+1), 0, size-1)
        x_coords = np.clip(cx + np.arange(-r, r+1), 0, size-1)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < size and 0 <= px < size:
                        img[py, px] = [
                            np.random.randint(80, 140),
                            np.random.randint(50, 100),
                            np.random.randint(120, 180),
                        ]

    return Image.fromarray(img)


def create_synthetic_xray(size=256, seed=None):
    """
    Create a synthetic grayscale image reminiscent of a chest X-ray.
    This is purely for UI demo — NOT a real X-ray.
    """
    if seed is not None:
        np.random.seed(seed)

    # Grayscale gradient with noise
    y_grad = np.linspace(0.3, 0.7, size)
    x_grad = np.linspace(0.4, 0.6, size)
    grid = np.outer(y_grad, x_grad)

    # Add some structure
    noise = np.random.normal(0, 0.05, (size, size))
    img_float = np.clip(grid + noise, 0, 1)

    # Oval shape (lung-like)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    oval = ((X - cx) / (size * 0.35))**2 + ((Y - cy) / (size * 0.45))**2
    img_float = np.where(oval < 1, img_float * 0.7, img_float * 0.95)

    img = (img_float * 255).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')


def main():
    # Create sample directories
    histo_dir = os.path.join(SAMPLES_DIR, 'histopathology_tumor_detection')
    xray_dir = os.path.join(SAMPLES_DIR, 'chest_xray_pneumonia')
    os.makedirs(histo_dir, exist_ok=True)
    os.makedirs(xray_dir, exist_ok=True)

    print("Generating synthetic sample images for demo mode...")
    print("NOTE: These are synthetic images for UI testing only.")
    print("      They are NOT real medical images.\n")

    # Histopathology samples
    for i in range(3):
        img = create_synthetic_histopathology_patch(256, seed=i * 100)
        path = os.path.join(histo_dir, f'sample_patch_{i+1}.png')
        img.save(path)
        print(f"  Created: {path}")

    # X-ray samples
    for i in range(3):
        img = create_synthetic_xray(256, seed=i * 200)
        path = os.path.join(xray_dir, f'sample_xray_{i+1}.png')
        img.save(path)
        print(f"  Created: {path}")

    print(f"\nDone. Samples saved to {SAMPLES_DIR}")
    print("These synthetic images are for UI demonstration only.")


if __name__ == '__main__':
    main()
