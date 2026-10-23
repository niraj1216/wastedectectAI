"""Optional FastAPI backend (pip install fastapi uvicorn python-multipart). Run: uvicorn api:app"""
import base64
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
import waste_core as wc

app = FastAPI(title="WasteDetect AI API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), conf: float = 0.2, include_image: bool = False):
    img = wc.load_image(await file.read())
    if img is None:
        raise HTTPException(400, "Could not decode image")
    annotated, a = wc.analyze(img, conf)
    out = {"items": len(a.detections), "counts": a.counts, "coverage": round(a.coverage, 4),
           "severity_score": a.score, "priority": a.level, "actions": a.actions,
           "detections": [{"label": d.label, "confidence": round(d.confidence, 3),
                           "box": [round(v) for v in d.box]} for d in a.detections]}
    if include_image:
        out["annotated_jpeg_base64"] = base64.b64encode(cv2.imencode(".jpg", annotated)[1]).decode()
    return out
