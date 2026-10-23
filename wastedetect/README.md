# WasteDetect AI

Drag & drop street photos -> litter detection (zero-shot YOLO-World via ONNX Runtime), 0-100 severity
score, cleanup priority and a downloadable report. Runs in under 512 MB RAM (no PyTorch at runtime).

## Deploy on Render (free)
1. Push this folder to GitHub (Dockerfile at repo root).
2. Render > New > Blueprint > pick the repo > Apply (reads render.yaml, free plan).
3. First build ~10-15 min (exports the model to ONNX in a throwaway stage). Open the onrender.com URL.

If the build can't export the model: run `pip install -r requirements-export.txt && python export_onnx.py`
(or in Colab), commit `models/waste_world.onnx` (~50 MB), and set `dockerfilePath: ./Dockerfile.prebuilt`.
If you hit out-of-memory at runtime: build with a smaller input size (`EXPORT_IMGSZ=512`).

## Local
```bash
pip install -r requirements.txt && pip install -r requirements-export.txt && python export_onnx.py
streamlit run app.py
```

## Score
`100 * (0.6*min(1, weighted_items/15) + 0.4*min(1, coverage/0.25))`; Low <25, Moderate <50, High <75, Critical >=75.
