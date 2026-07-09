"""
Data loading for the crop disease detection project.

Expects DATA_DIR laid out as:
    data/raw/<ClassName>/<image>.jpg
    data/raw/<ClassName>/<image>.jpg
    ...
one subfolder per class, which is exactly how the PlantVillage dataset
(and Kaggle's "New Plant Diseases Dataset" mirror of it) ships.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import tensorflow as tf
from tensorflow import keras


def get_datasets(data_dir=None, img_size=None, batch_size=None,
                  val_split=None, seed=None):
    """Build train/val tf.data.Dataset objects from a directory of class folders."""
    data_dir = data_dir or config.DATA_DIR
    img_size = img_size or config.IMG_SIZE
    batch_size = batch_size or config.BATCH_SIZE
    val_split = val_split if val_split is not None else config.VAL_SPLIT
    seed = seed or config.SEED

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run scripts/download_data.py first, or point CROP_DATA_DIR at "
            f"your own copy of the dataset."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )
    # NOTE: `shuffle` must match train_ds's value (both default to True) so the
    # two calls shuffle the master file list the same way before splitting --
    # otherwise train/val come from inconsistent orderings, which risks
    # overlap between them and can skew which classes land in validation.
    # The shared `seed` still makes this split reproducible across runs.
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
