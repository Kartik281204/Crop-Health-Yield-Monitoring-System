# Crop Disease Detection

A CNN that classifies crop leaf images as healthy or diseased (naming the
specific disease), covering 4 crops and 8 classes in this initial version:
Apple (healthy / black rot), Corn (healthy / common rust), Potato (healthy /
early blight), and Tomato (healthy / late blight).

Inspired by [Shubham Jain's Crop-Disease-Detection](https://github.com/Shubham-Jain-09/Crop-Disease-Detection),
which tackles the same problem with a frozen pretrained AlexNet backbone
across the full 38-class PlantVillage dataset. This project is an
independent implementation, not a fork -- built from scratch with a
different architecture and training approach.

## Results (this run)

Trained on a balanced ~2,000-image subset (≈250 images/class) of the
[PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset),
80/20 train/val split.

| Metric | Value |
|---|---|
| Validation accuracy | **80.2%** |
| Validation loss | 0.667 |
| Classes | 8 |
| Training images | 1,578 |
| Validation images | 394 |

Per-class detail is in [`outputs/classification_report.txt`](outputs/classification_report.txt)
and [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png); training
curves are in [`outputs/training_curves.png`](outputs/training_curves.png).

**Where it's strong:** cross-crop identification is essentially solved here
-- Corn (rust vs. healthy) scores 98-99% F1. **Where it struggles:**
fine-grained within-crop calls, specifically Tomato late blight vs. Tomato
healthy (37% F1 on late blight) -- the model tends to call borderline late
blight leaves "healthy". That's a real limitation worth naming rather than
hiding, and the likely fix is more late-blight examples plus finer-grained
augmentation, not a bigger model.

## Why a from-scratch CNN instead of transfer learning

`src/model.py` tries `MobileNetV2` + ImageNet weights first -- that's the
better path (fewer images needed, higher accuracy, faster to train) and
what it'll use automatically anywhere with normal internet access, e.g.
Colab or Kaggle. It only falls back to a compact from-scratch CNN
(3 conv blocks, heavy L2 + dropout + augmentation, no BatchNorm -- see note
below) when the pretrained weights can't be downloaded. The results above
are from the **fallback path**, trained in a network-restricted sandbox
with no GPU and no access to the weights host. Re-running this on a machine
with open internet access should meaningfully beat these numbers.

One implementation detail worth flagging if you extend this: an earlier
version used BatchNorm in the scratch CNN, but with only ~50 steps/epoch on
this little data, BatchNorm's running statistics never stabilized, and
validation accuracy sat at random chance while training accuracy climbed --
a classic small-data/few-steps BatchNorm trap. Removing it fixed it.

## Project structure

```
crop-disease-detection/
├── src/
│   ├── config.py      # all hyperparameters and paths live here
│   ├── dataset.py      # loads data/raw/<ClassName>/*.jpg into train/val splits
│   ├── model.py        # MobileNetV2 transfer learning + from-scratch fallback
│   ├── train.py         # trains one epoch per run, checkpoints, resumes automatically
│   ├── evaluate.py      # classification report + confusion matrix on the val split
│   └── predict.py       # single-image inference from the command line
├── scripts/
│   └── download_data.py # pulls a balanced class subset from the PlantVillage GitHub mirror
├── models/               # crop_disease_model.keras + class_names.json (committed -- see below)
├── outputs/              # training_curves.png, confusion_matrix.png, classification_report.txt
├── data/raw/              # gitignored -- regenerate with scripts/download_data.py
└── requirements.txt
```

The trained model (`models/crop_disease_model.keras`, ~800KB) is committed
so the repo is demoable without retraining. Training data is not committed
(see `.gitignore`) -- regenerate it with the download script below.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py      # ~2,000 images, ~35MB
cd src
python train.py                       # one epoch per run -- see below
```

`train.py` trains exactly one epoch, checkpoints, and exits, so it's safe
to interrupt and resume. Loop it until it prints `TRAINING COMPLETE`:

```bash
for i in $(seq 1 40); do python train.py; done
```

Then:

```bash
python evaluate.py                    # classification report + confusion matrix
python predict.py /path/to/leaf.jpg   # single-image prediction
```

## Scaling up

This ships with 8 of the full dataset's 38 classes and ~250 images/class,
sized to train reasonably fast on a single CPU core. To scale to the real
thing:

1. Add more classes to `CLASSES` in `scripts/download_data.py` (full list
   in the [PlantVillage repo](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)),
   or point `CROP_DATA_DIR` at a full copy of the dataset (e.g. Kaggle's
   ["New Plant Diseases Dataset"](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset),
   which mirrors PlantVillage).
2. Run it somewhere with normal internet access so the MobileNetV2 +
   ImageNet path actually engages -- expect meaningfully better accuracy
   and much less tuning than the from-scratch fallback.
3. Ideally use a GPU (Colab/Kaggle are free options) -- full 38-class
   training from scratch on CPU would take a long time.
4. Classes are imbalanced in the full dataset (some have 150 images,
   others 5,000+); consider `class_weight` in `model.fit` or oversampling
   the smaller classes.
5. Set `config.FINE_TUNE = True` to unfreeze the top of the MobileNetV2
   backbone for a short low-LR pass after the initial frozen-backbone
   training -- usually good for a few extra points of accuracy.

## Acknowledgments

- Dataset: Hughes & Salathé, ["An open access repository of images on
  plant health..."](https://arxiv.org/abs/1511.08060) (PlantVillage),
  via the [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
  GitHub mirror.
- Inspiration: [Shubham Jain's Crop-Disease-Detection](https://github.com/Shubham-Jain-09/Crop-Disease-Detection).
