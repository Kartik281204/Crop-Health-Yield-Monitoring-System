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

## Phase 2: serving API

A FastAPI service wraps the trained model behind a REST API, with a small
static test page so you can try it in a browser without curl.

```
api/
├── main.py       # routes: /health, /classes, /predict
├── inference.py   # bridges the API to src/predict.py -- one prediction
│                   #   implementation, not two
├── schemas.py      # Pydantic response models
└── static/
    └── index.html   # drag-and-drop test page, served at /
tests/
└── test_api.py       # 7 tests via FastAPI's TestClient, no server needed
```

Run it:

```bash
cd api
uvicorn main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/` for the test page, or
`http://127.0.0.1:8000/docs` for interactive Swagger docs (auto-generated
by FastAPI from the Pydantic schemas).

**Endpoints:**

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status, model_loaded, num_classes}` |
| GET | `/classes` | `{classes: [...]}` |
| POST | `/predict` | `{predicted_class, confidence, top_3: [...]}` -- send a JPEG/PNG/WebP as `file` in multipart form data |

The model loads once at startup (not per-request) via FastAPI's `lifespan`
hook. `/predict` validates content-type and file size (8MB cap) before
doing anything expensive, and returns a 400 with a clear `detail` message
for anything it rejects rather than a raw 500.

Run the tests:

```bash
pip install -r requirements.txt
pytest tests/test_api.py -v
```

All 7 pass against the current model: health/classes shape, a real
prediction end-to-end, and three rejection cases (wrong content-type,
empty file, corrupt bytes).

## Phase 3: frontend

A Next.js app replaces the Phase 2 static test page with a real product
surface -- same visual language (dark leaf-green/rust palette, monospace
data readouts), now with drag-and-drop upload, a click-to-try sample
gallery, and live confidence bars.

```
frontend/
├── app/
│   ├── layout.tsx      # root layout -- system font stack, no build-time
│   │                     #   Google Fonts fetch (see note below)
│   ├── page.tsx          # composes everything below
│   └── globals.css        # Tailwind v4 theme -- palette as CSS variables
├── components/
│   ├── UploadZone.tsx      # drag-and-drop + click-to-browse
│   ├── ExampleGallery.tsx   # 8 sample leaves, one per class, click to test
│   ├── ResultCard.tsx        # prediction + confidence bars
│   └── StatusPill.tsx         # live /health poll in the footer
├── lib/
│   └── api.ts                  # typed fetch wrapper around the Phase 2 API
├── public/samples/               # the 8 example images ExampleGallery uses
└── .env.local.example              # NEXT_PUBLIC_API_URL
```

Run it (with the API from Phase 2 running separately on port 8000):

```bash
cd frontend
npm install
cp .env.local.example .env.local   # point this at your API if not localhost:8000
npm run dev
```

Open `http://localhost:3000`.

**Verified, not just built:** `npm run build` compiles clean (TypeScript +
ESLint, zero errors). With both the API and the frontend running
together, a real `POST /predict` from the frontend's origin returns
`200` with `access-control-allow-origin: *` and a valid prediction --
confirmed the CORS setup from Phase 2 actually works cross-origin, not
just from `curl`.

One deliberate choice worth flagging: `layout.tsx` skips `next/font/google`
(the framework's usual font approach) in favor of a plain system-font
stack. `next/font` fetches from Google Fonts at *build* time, which fails
anywhere without open internet access -- same class of problem as the
ImageNet weights in Phase 1. System fonts sidestep it entirely and cost
zero extra requests; swap in `next/font` later if you want a specific
custom typeface once this is running somewhere unrestricted.

## Phase 4: containers + CI

One-command local stack, and tests that run automatically on push.

```
├── docker-compose.yml
├── .dockerignore
├── api/Dockerfile           # python:3.12-slim, libgomp1 for TF, healthcheck
├── frontend/Dockerfile       # node:22-slim, 3-stage build, standalone output
├── frontend/.dockerignore
├── requirements.txt            # trimmed to runtime-only deps (Docker uses this)
├── requirements-dev.txt         # + pytest/httpx, for local dev and CI
└── .github/workflows/ci.yml       # pytest job + npm build/lint job
```

```bash
docker compose up --build
```
brings up the API on `:8000` and the frontend on `:3000` together.

**Important if you deploy this anywhere other than localhost:** the
frontend calls the API from the *browser* (client-side `fetch`), so
`NEXT_PUBLIC_API_URL` has to be an address the browser can reach --
`docker-compose.yml` bakes in `http://localhost:8000`, which only works
when both containers' ports are mapped to your own machine. Deploying
API and frontend to different hosts means passing the real public API
URL as a build arg instead.

**Honesty check, since this matters:** this sandbox has no Docker
installed, so the two Dockerfiles and `docker-compose.yml` are
carefully written and logic-reviewed, but **not build-tested** -- unlike
everything in Phases 1-3, which was actually run. What I could verify,
and did: the exact `npm run build` the frontend Dockerfile invokes
succeeds and produces the `server.js` + `public/` + `.next/static/`
layout the Dockerfile copies; `npm ci` installs cleanly from the
committed lockfile; and the test suite (now using a generated in-memory
image for the core predict test, not a downloaded file) passes without
needing `data/raw` to exist -- which matters because CI checks out a
clean repo with no gitignored data in it. Worth a `docker compose up
--build` on your end before trusting it blindly.

**CI** (`.github/workflows/ci.yml`) runs on every push/PR to `main`:
one job installs `requirements-dev.txt` and runs `pytest`, another runs
`npm ci`, `npm run lint`, and `npm run build` for the frontend. Both
jobs run the same commands already verified above, just in GitHub's
infrastructure instead of this sandbox.

## Roadmap

- ~~Phase 1: trained model~~ -- done, 80.2% val accuracy (8 classes)
- ~~Phase 2: FastAPI serving layer~~ -- done, tested end-to-end
- ~~Phase 3: Next.js frontend~~ -- done, verified against a live API with real CORS
- ~~Phase 4: Docker + CI~~ -- written and reviewed; build-test with `docker compose up --build` before relying on it
- Later: full 38-class dataset + real transfer learning on GPU (see "Scaling up" above), then redeploy

## Acknowledgments

- Dataset: Hughes & Salathé, ["An open access repository of images on
  plant health..."](https://arxiv.org/abs/1511.08060) (PlantVillage),
  via the [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
  GitHub mirror.
- Inspiration: [Shubham Jain's Crop-Disease-Detection](https://github.com/Shubham-Jain-09/Crop-Disease-Detection).
