"""
Run inference on a single image.

Usage:
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


def predict_image(image_path, model=None, class_names=None):
    if model is None:
        model = keras.models.load_model(config.MODEL_PATH)
    if class_names is None:
        class_names = json.load(open(config.CLASS_NAMES_PATH))

    img = keras.utils.load_img(image_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]
    return [(class_names[i], float(preds[i])) for i in order]


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
