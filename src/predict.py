"""
Prediction logic, shared by the CLI below and the FastAPI service in api/.

Usage (CLI):
    cd src
    python predict.py /path/to/leaf.jpg
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import numpy as np
from tensorflow import keras

_model = None
_class_names = None


def load_model_and_classes():
    """Load once, reuse everywhere (CLI and API both call this)."""
    global _model, _class_names
    if _model is None:
        _model = keras.models.load_model(config.MODEL_PATH)
        _class_names = json.load(open(config.CLASS_NAMES_PATH))
    return _model, _class_names


def preprocess_pil_image(pil_img):
    """PIL Image -> model-ready batch of 1. Works for any source (disk, upload, camera)."""
    pil_img = pil_img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = keras.utils.img_to_array(pil_img)
    return np.expand_dims(arr, axis=0)


def predict_array(arr, model, class_names):
    """Core prediction step: preprocessed array in, ranked (label, confidence) list out."""
    preds = model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]
    return [(class_names[i], float(preds[i])) for i in order]


def predict_image(image_path, model=None, class_names=None):
    """CLI/file-path entry point."""
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()
    img = keras.utils.load_img(image_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return predict_array(arr, model, class_names)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    ranked = predict_image(image_path)
    top_label, top_conf = ranked[0]
    print(f"Prediction: {top_label}  ({top_conf * 100:.1f}% confidence)")
    print("Top-3:")
    for label, conf in ranked[:3]:
        print(f"  {label}: {conf * 100:.1f}%")


if __name__ == "__main__":
    main()
