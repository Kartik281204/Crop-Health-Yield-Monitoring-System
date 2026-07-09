"""
Crop Disease Detection API.

Run it:
    cd api
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/ for the test page, or
http://127.0.0.1:8000/docs for interactive API docs.
"""
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import warm_up, predict_image_bytes
from schemas import (
    PredictionResponse,
    ClassConfidence,
    HealthResponse,
    ClassesResponse,
)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024  # 8MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()  # load the model once, at startup, not on the first request
    yield


app = FastAPI(
    title="Crop Disease Detection API",
    description="Upload a crop leaf photo, get a disease prediction back.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    model, class_names = warm_up()
    return HealthResponse(
        status="ok", model_loaded=model is not None, num_classes=len(class_names)
    )


@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes():
    _, class_names = warm_up()
    return ClassesResponse(classes=class_names)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Use JPEG, PNG, or WebP.",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 8MB).")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        ranked = predict_image_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    top_label, top_conf = ranked[0]
    return PredictionResponse(
        predicted_class=top_label,
        confidence=top_conf,
        top_3=[ClassConfidence(label=l, confidence=c) for l, c in ranked[:3]],
    )


# Mounted last on purpose: this is a catch-all for "/", so the explicit
# routes above (/health, /classes, /predict) must be registered first or
# this would shadow them.
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
