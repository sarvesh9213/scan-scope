"""
Input validation for uploaded images.

Validates file type, file size, image readability, and basic
compatibility before any model is invoked.
"""
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image

import config


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    image_info: Optional[dict] = None


def validate_file_extension(filename: str) -> ValidationResult:
    """Check that the file has an allowed extension."""
    if '.' not in filename:
        return ValidationResult(
            is_valid=False,
            error_message="File has no extension. Supported formats: "
                          f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}."
        )
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            error_message=f"File type '.{ext}' is not supported. "
                          f"Supported formats: "
                          f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}."
        )
    return ValidationResult(is_valid=True)


def validate_file_size(file_path: str) -> ValidationResult:
    """Check that the file is within the size limit."""
    size = os.path.getsize(file_path)
    max_mb = config.MAX_CONTENT_LENGTH / (1024 * 1024)
    if size > config.MAX_CONTENT_LENGTH:
        actual_mb = size / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=f"File is too large ({actual_mb:.1f} MB). "
                          f"Maximum allowed size is {max_mb:.0f} MB."
        )
    if size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File is empty (0 bytes)."
        )
    return ValidationResult(is_valid=True)


def validate_image_readable(file_path: str) -> ValidationResult:
    """Check that the file is a valid, readable image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open after verify (verify can close the file)
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format

        info = {
            'width': width,
            'height': height,
            'mode': mode,
            'format': fmt,
        }

        warning = None
        if width < 10 or height < 10:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image is too small ({width}×{height} pixels). "
                              "Please upload a larger image."
            )

        if width > 10000 or height > 10000:
            warning = (
                f"Image is very large ({width}×{height} pixels). "
                "It will be resized for model input, which may affect results."
            )

        return ValidationResult(
            is_valid=True,
            warning_message=warning,
            image_info=info,
        )

    except (IOError, SyntaxError, Exception) as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Could not read this file as an image: {str(e)}. "
                          "Please upload a valid image file."
        )


def validate_upload(file_path: str, filename: str) -> ValidationResult:
    """
    Run all validations on an uploaded file.

    Returns the first failing validation, or a success result with
    combined image info and warnings.
    """
    # Check extension
    result = validate_file_extension(filename)
    if not result.is_valid:
        return result

    # Check size
    result = validate_file_size(file_path)
    if not result.is_valid:
        return result

    # Check image readability
    result = validate_image_readable(file_path)
    return result
