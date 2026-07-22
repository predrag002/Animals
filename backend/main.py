import io
import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn



CLASSES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 128

MODELS_DIR  = Path(os.getenv("MODELS_DIR", "/app/models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/app/results"))



class TransferMobileNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        backbone      = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(1280, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout/2), nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.pool(self.features(x)))






TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])



def load_best_model():
    checkpoint = None
    for fold in range(1, 6):
        ckpt = MODELS_DIR / f"Config4_MobileNet_finetune_fold{fold}.pt"
        if ckpt.exists():
            checkpoint = ckpt
            break

    if checkpoint is None:
        raise FileNotFoundError(
            f"Nije pronadjen Config4_MobileNet_finetune_fold*.pt u {MODELS_DIR}."
        )

    print(f"Checkpoint: {checkpoint}")
    model = TransferMobileNet(num_classes=NUM_CLASSES)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, "TransferMobileNet", str(checkpoint.name)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Animals-10 Klasifikator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalni model
MODEL      = None
MODEL_NAME = None
CKPT_NAME  = None


@app.on_event("startup")
async def startup():
    global MODEL, MODEL_NAME, CKPT_NAME
    try:
        MODEL, MODEL_NAME, CKPT_NAME = load_best_model()
        print(f"Model spreman: {MODEL_NAME}")
    except Exception as e:
        print(f"UPOZORENJE: {e}")
        print("API radi ali /predict nece raditi bez modela.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":      MODEL_NAME,
        "checkpoint": CKPT_NAME,
        "model_loaded": MODEL is not None,
    }


@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Model nije ucitan. Proveri models/ folder.")

    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fajl mora biti slika (jpg, png...)")

    # Ucitaj i transformisi sliku
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Ne mogu da otvorim sliku.")

    tensor = TRANSFORM(img).unsqueeze(0)  # [1, 3, H, W]

    
    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    
    top3_vals, top3_idx = probs.topk(3)
    top3 = [
        {"class": CLASSES[i], "confidence": round(float(p), 4)}
        for i, p in zip(top3_idx.tolist(), top3_vals.tolist())
    ]

    return {
        "prediction":  top3[0]["class"],
        "confidence":  top3[0]["confidence"],
        "top3":        top3,
        "model_used":  MODEL_NAME,
    }



frontend_path = Path("/app/frontend")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)