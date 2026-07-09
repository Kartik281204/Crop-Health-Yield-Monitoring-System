"""
API tests -- run from the project root:
    pytest tests/test_api.py -v

Uses FastAPI's TestClient, so no server needs to be running.
"""
import io
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

VALID_CLASSES = [
    "Apple_Black_rot", "Apple_healthy", "Corn_Common_rust", "Corn_healthy",
    "Potato_Early_blight", "Potato_healthy", "Tomato_Late_blight", "Tomato_healthy",
]


def _synthetic_image_bytes():
    """A random-noise RGB image, generated in memory. Used instead of a
    real leaf photo for the core predict test so it doesn't depend on
    data/raw -- which is gitignored and won't exist on a fresh checkout
    (e.g. in CI, before anyone's run scripts/download_data.py)."""
    img = Image.effect_noise((256, 256), 60).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _any_real_sample_image():
    matches = glob.glob(os.path.join(DATA_DIR, "*", "*"))
    if not matches:
        pytest.skip("No sample images in data/raw -- run scripts/download_data.py first.")
    return matches[0]


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] == 8


def test_classes():
    res = client.get("/classes")
    assert res.status_code == 200
    classes = res.json()["classes"]
    assert len(classes) == 8
    assert "Tomato_healthy" in classes


def test_predict_valid_image():
    """Core end-to-end check -- runs everywhere, including a fresh CI
    checkout with no downloaded dataset. Asserts response shape/validity,
    not prediction correctness (a noise image has no "right" class)."""
    res = client.post(
        "/predict",
        files={"file": ("leaf.jpg", _synthetic_image_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] in VALID_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_3"]) == 3
    confidences = [c["confidence"] for c in body["top_3"]]
    assert confidences == sorted(confidences, reverse=True)  # descending


def test_predict_real_sample_if_available():
    """Extra sanity check for local dev: same assertions, but against a
    real downloaded leaf photo instead of noise. Skips cleanly if
    data/raw hasn't been populated (that's expected/fine in CI)."""
    path = _any_real_sample_image()
    with open(path, "rb") as f:
        res = client.post("/predict", files={"file": ("leaf.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] in VALID_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_bad_content_type():
    res = client.post(
        "/predict", files={"file": ("notes.txt", b"hello world", "text/plain")}
    )
    assert res.status_code == 400
    assert "Unsupported content type" in res.json()["detail"]


def test_predict_rejects_empty_file():
    res = client.post(
        "/predict", files={"file": ("leaf.jpg", b"", "image/jpeg")}
    )
    assert res.status_code == 400


def test_predict_rejects_corrupt_image_bytes():
    # Valid content-type header, but the bytes aren't a real image.
    res = client.post(
        "/predict", files={"file": ("leaf.jpg", b"not actually a jpeg", "image/jpeg")}
    )
    assert res.status_code == 400
    assert "Not a readable image" in res.json()["detail"]


def test_static_page_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Crop Diagnostic" in res.text
