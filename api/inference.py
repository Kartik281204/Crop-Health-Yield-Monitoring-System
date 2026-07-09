"""
Thin bridge between the API layer and src/predict.py, so there's exactly
one implementation of "load the model" and "run a prediction" -- the CLI
and the API both call into src/, neither duplicates the logic.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predict import load_model_and_classes, preprocess_pil_image, predict_array

from PIL import Image, UnidentifiedImageError


def warm_up():
    """Call once at API startup so the first real request isn't the one
    paying for model load time."""
    return load_model_and_classes()


def predict_image_bytes(image_bytes):
    """Raw uploaded bytes -> ranked [(label, confidence), ...]. Raises
    ValueError on anything that isn't a readable image."""
    model, class_names = load_model_and_classes()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Not a readable image: {e}")

    arr = preprocess_pil_image(pil_img)
    return predict_array(arr, model, class_names)
