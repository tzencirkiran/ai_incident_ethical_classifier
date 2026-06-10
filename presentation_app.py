"""FastAPI presentation server for the AI ethics incident classifier."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model.infer import CHECKPOINT_DIR, DEVICE, MAX_LENGTH, build_text, load_model


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web"


app = FastAPI(
    title="AI Ethics Incident Classifier",
    description="Presentation API for multi-label AI ethics incident classification.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IncidentRequest(BaseModel):
    headline: str = Field(..., min_length=1, max_length=500)
    purpose: Optional[str] = Field(default=None, max_length=300)
    technology: Optional[str] = Field(default=None, max_length=300)
    deployer: Optional[str] = Field(default=None, max_length=300)
    developer: Optional[str] = Field(default=None, max_length=300)
    system_name: Optional[str] = Field(default=None, max_length=300)
    news_trigger: Optional[str] = Field(default=None, max_length=300)
    jurisdiction: Optional[str] = Field(default=None, max_length=300)
    sector: Optional[str] = Field(default=None, max_length=300)


@lru_cache(maxsize=1)
def get_artifacts():
    """Load the model once per server process."""
    return load_model()


def predict_scores(text):
    tokenizer, model, label_classes, thresholds = get_artifacts()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**encoding).logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    scores = []
    for label, probability, threshold in zip(label_classes, probabilities, thresholds):
        prob = float(probability)
        cutoff = float(threshold)
        scores.append(
            {
                "label": str(label),
                "probability": prob,
                "threshold": cutoff,
                "selected": prob >= cutoff,
            }
        )

    return sorted(scores, key=lambda item: item["probability"], reverse=True)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    checkpoint = Path(CHECKPOINT_DIR)
    required_files = [
        "config.json",
        "label_classes.npy",
        "model.safetensors",
        "thresholds.npy",
        "tokenizer_config.json",
        "vocab.txt",
    ]
    missing = [name for name in required_files if not (checkpoint / name).exists()]
    return {
        "ok": not missing,
        "checkpoint_dir": str(checkpoint),
        "missing": missing,
        "device": str(DEVICE),
    }


@app.get("/api/labels")
def labels():
    _, _, label_classes, thresholds = get_artifacts()
    return {
        "labels": [
            {"label": str(label), "threshold": float(threshold)}
            for label, threshold in zip(label_classes, thresholds)
        ]
    }


@app.post("/api/predict")
def predict(request: IncidentRequest):
    headline = request.headline.strip()
    if not headline:
        raise HTTPException(status_code=422, detail="Headline cannot be empty.")

    text = build_text(
        headline,
        purpose=request.purpose,
        technology=request.technology,
        deployer=request.deployer,
        developer=request.developer,
        system_name=request.system_name,
        news_trigger=request.news_trigger,
        jurisdiction=request.jurisdiction,
        sector=request.sector,
    )
    scores = predict_scores(text)
    predictions = [item for item in scores if item["selected"]]
    return {
        "input_text": text,
        "predictions": predictions,
        "scores": scores,
    }
