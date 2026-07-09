"""
Train the crop disease classifier -- one epoch per invocation.

This trains a single epoch, checkpoints, and exits. Run it repeatedly
(e.g. `for i in $(seq 1 12); do python train.py; done`) and it resumes
from where it left off each time, stopping early on its own once
validation loss stops improving. This keeps each run short and safe to
interrupt, rather than needing one long-lived process.

Usage:
    cd src
    python train.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import get_datasets
from model import build_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

PATIENCE = 5


def load_state():
    if os.path.exists(config.TRAIN_STATE_PATH):
        return json.load(open(config.TRAIN_STATE_PATH))
    return {
        "epoch": 0,
        "history": {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []},
        "best_val_loss": None,
        "best_epoch": 0,
        "patience_count": 0,
        "done": False,
    }


def save_state(state):
    with open(config.TRAIN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def plot_history(history, out_path):
    epochs = range(1, len(history["accuracy"]) + 1)
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="train")
    plt.plot(epochs, history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    state = load_state()

    if state["done"]:
        print("TRAINING COMPLETE (already finished on a previous run).")
        return

    print(f"Loading data from {config.DATA_DIR} ...")
    train_ds, val_ds, class_names = get_datasets()

    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"Resuming from checkpoint, epoch {state['epoch']} completed so far.")
        model = keras.models.load_model(config.CHECKPOINT_PATH)
    else:
        print("No checkpoint found -- building a fresh model.")
        model, _base_model = build_model(num_classes=len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)
    dt = time.time() - t0

    for k, v in hist.history.items():
        state["history"][k].append(float(v[0]))
    state["epoch"] += 1

    model.save(config.CHECKPOINT_PATH)

    val_loss = hist.history["val_loss"][0]
    improved = state["best_val_loss"] is None or val_loss < state["best_val_loss"] - 1e-4
    if improved:
        state["best_val_loss"] = val_loss
        state["best_epoch"] = state["epoch"]
        state["patience_count"] = 0
        model.save(config.MODEL_PATH)
        tag = "(new best, saved as main model)"
    else:
        state["patience_count"] += 1
        tag = f"(no improvement, {state['patience_count']}/{PATIENCE})"

    print(
        f"Epoch {state['epoch']}/{config.EPOCHS} done in {dt:.1f}s -- "
        f"train_acc={hist.history['accuracy'][0]:.3f} val_acc={hist.history['val_accuracy'][0]:.3f} "
        f"val_loss={val_loss:.3f} {tag}"
    )

    finished = state["epoch"] >= config.EPOCHS or state["patience_count"] >= PATIENCE
    if finished:
        state["done"] = True
        plot_history(state["history"], os.path.join(config.OUTPUT_DIR, "training_curves.png"))
        with open(os.path.join(config.OUTPUT_DIR, "training_history.json"), "w") as f:
            json.dump(state["history"], f, indent=2)
        print(
            f"TRAINING COMPLETE. Best epoch={state['best_epoch']} "
            f"best_val_loss={state['best_val_loss']:.3f}. Model at {config.MODEL_PATH}"
        )
    else:
        print("CONTINUE: run train.py again for the next epoch.")

    save_state(state)


if __name__ == "__main__":
    main()
