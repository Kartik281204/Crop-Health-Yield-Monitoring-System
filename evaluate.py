"""
Evaluate the trained model on the held-out validation split:
classification report (precision/recall/F1 per class) + confusion matrix.

Usage:
    cd src
    python evaluate.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import get_datasets

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def main():
    print(f"Loading model from {config.MODEL_PATH} ...")
    model = keras.models.load_model(config.MODEL_PATH)
    class_names = json.load(open(config.CLASS_NAMES_PATH))

    _, val_ds, _ = get_datasets()

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    print(report)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    plt.figure(figsize=(1.1 * n + 2, 1.1 * n + 2))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(n), class_names, rotation=90)
    plt.yticks(range(n), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (validation split)")
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            plt.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8,
            )
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved classification report + confusion matrix to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
