"""
Model architecture: MobileNetV2 transfer learning.

Why MobileNetV2 instead of training a CNN from scratch (or freezing an
older AlexNet backbone, which is the more common approach in older
plant-disease tutorials): it's a fully-convolutional, depthwise-separable
architecture designed to be cheap to run, which matters a lot on a
single CPU core, and its ImageNet features transfer well to leaf imagery
since both share texture/edge/color statistics.

Augmentation and preprocessing are baked directly into the model graph,
so the exported .keras file is a single self-contained artifact --
predict.py just loads it and feeds raw images straight in, no separate
preprocessing step to remember or get out of sync.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_scratch_cnn(num_classes, img_size):
    """A compact, heavily-regularized CNN for training from scratch on a
    few hundred images per class -- used automatically when ImageNet
    weights can't be downloaded (e.g. a network-restricted sandbox).
    Kept intentionally small (vs. e.g. 4 blocks up to 128 filters) and
    L2/dropout-heavy, since a bigger network memorizes a dataset this
    size almost immediately."""
    reg = keras.regularizers.l2(1e-4)

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (24, 48, 96):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                           kernel_regularizer=reg)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(0.1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(96, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="crop_disease_scratch_cnn")


def build_model(num_classes, img_size=None):
    """
    Tries MobileNetV2 + ImageNet transfer learning first (the recommended
    path -- fast to train, strong accuracy, and what you should use once
    this is running somewhere with normal internet access, e.g. Colab or
    Kaggle). Falls back to a compact from-scratch CNN if the pretrained
    weights can't be downloaded, which happens in network-restricted
    environments -- this keeps the pipeline runnable everywhere, just
    with a clear accuracy trade-off in the fallback case.
    """
    img_size = img_size or config.IMG_SIZE

    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # frozen for the initial training pass

        inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
        x = build_augmentation()(inputs)
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")
        print("Using MobileNetV2 + ImageNet transfer learning.")
        return model, base_model

    except Exception as e:
        print(
            f"[fallback] Couldn't download ImageNet weights ({type(e).__name__}: {e}). "
            f"Training a compact CNN from scratch instead. This still works, but expect "
            f"noticeably lower accuracy than transfer learning -- re-run on a machine "
            f"with normal internet access (Colab/Kaggle) to use the MobileNetV2 path."
        )
        model = build_scratch_cnn(num_classes, img_size)
        return model, None


def unfreeze_for_fine_tuning(base_model, fine_tune_at):
    """Unfreeze the top of the backbone for a short low-LR fine-tuning pass."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model
