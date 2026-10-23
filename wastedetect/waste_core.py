"""WasteDetect AI core: ONNX inference (no PyTorch), severity scoring, report. Low-RAM by design."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import cv2
import numpy as np
from prompts import WORLD_PROMPTS

DEFAULT_MODEL = os.getenv("WASTE_ONNX", os.path.join(os.path.dirname(__file__), "models", "waste_world.onnx"))
CLASS_KEYS = list(WORLD_PROMPTS)
MAX_SIDE = 1280  # uploads are downscaled to this to cap memory

ITEMS_FOR_FULL_SCORE = 15.0
COVERAGE_FOR_FULL_SCORE = 0.25
COUNT_WEIGHT, COVERAGE_WEIGHT = 0.6, 0.4

LEVELS = [(25, "Low", "#2e9e5b"), (50, "Moderate", "#e0b000"),
          (75, "High", "#e8742a"), (101, "Critical", "#d62f2f")]
ACTIONS = {
    "Low": ["Add to routine sweep schedule.", "Place or check litter bins nearby.",
            "Re-scan in 7 days to confirm the area stays clean."],
    "Moderate": ["Schedule a crew pickup within 48 hours.", "Check bin capacity and pickup frequency.",
                 "Flag the spot for the weekly litter report."],
    "High": ["Dispatch a cleanup crew within 24 hours.", "Bring bags for mixed waste and sort recyclables.",
             "Investigate recurring dumping or an overflowing bin."],
    "Critical": ["Escalate: same-day cleanup with a vehicle.", "Inspect for hazardous or bulky waste before handling.",
                 "Notify the municipal authority and consider signage or enforcement."],
}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple
    weight: float = 1.0


@dataclass
class Analysis:
    detections: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    weighted_items: float = 0.0
    coverage: float = 0.0
    score: float = 0.0
    level: str = "Low"
    color: str = "#2e9e5b"
    actions: list = field(default_factory=list)


_sessions = {}


def load_model(path: str = DEFAULT_MODEL):
    if path not in _sessions:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python export_onnx.py` to create it.")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False   # big RAM saver
        so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _sessions[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _sessions[path]


def load_image(data: bytes, max_side: int = MAX_SIDE):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def postprocess(out: np.ndarray, r: float, left: int, top: int, shape: tuple,
                conf: float, iou: float = 0.5) -> list:
    """Decode a YOLOv8-style raw output (1, 4+nc, N) into Detection objects (with NMS)."""
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                        # -> (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    sc = scores_all.max(1)
    keep = sc >= conf
    pred, cls, sc = pred[keep], cls[keep], sc[keep]
    if len(sc) == 0:
        return []
    h, w = shape[:2]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = np.clip((cx - bw / 2 - left) / r, 0, w)
    y1 = np.clip((cy - bh / 2 - top) / r, 0, h)
    x2 = np.clip((cx + bw / 2 - left) / r, 0, w)
    y2 = np.clip((cy + bh / 2 - top) / r, 0, h)
    boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = cv2.dnn.NMSBoxes(boxes, sc.astype(float).tolist(), conf, iou)
    dets = []
    for i in np.array(idx).flatten():
        key = CLASS_KEYS[int(cls[i])] if int(cls[i]) < len(CLASS_KEYS) else "item"
        label, weight = WORLD_PROMPTS.get(key, ("item", 1.0))
        dets.append(Detection(label, float(sc[i]), (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])), weight))
    return dets


def detect(image_bgr: np.ndarray, conf: float = 0.2, model_path: str = DEFAULT_MODEL) -> list:
    sess = load_model(model_path)
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
    canvas, r, left, top = letterbox(image_bgr, size)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {inp.name: np.ascontiguousarray(blob)})[0]
    return postprocess(out, r, left, top, image_bgr.shape, conf)


def coverage_fraction(dets: list, shape: tuple) -> float:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = 1
    return float(mask.sum()) / float(h * w) if h * w else 0.0


def severity(dets: list, shape: tuple) -> Analysis:
    """score = 100 * (0.6 * min(1, weighted_items/15) + 0.4 * min(1, coverage/0.25))"""
    a = Analysis(detections=dets)
    for d in dets:
        a.counts[d.label] = a.counts.get(d.label, 0) + 1
    a.weighted_items = sum(d.weight * d.confidence for d in dets)
    a.coverage = coverage_fraction(dets, shape)
    count_c = min(1.0, a.weighted_items / ITEMS_FOR_FULL_SCORE)
    cov_c = min(1.0, a.coverage / COVERAGE_FOR_FULL_SCORE)
    a.score = round(100 * (COUNT_WEIGHT * count_c + COVERAGE_WEIGHT * cov_c), 1)
    for limit, name, color in LEVELS:
        if a.score < limit:
            a.level, a.color = name, color
            break
    a.actions = ACTIONS[a.level]
    return a


def annotate(image_bgr: np.ndarray, dets: list) -> np.ndarray:
    out = image_bgr.copy()
    scale = max(out.shape[:2]) / 900
    thick = max(1, int(round(2 * scale)))
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 60, 230), thick)
        text = f"{d.label} {d.confidence:.0%}"
        fs = 0.5 * max(scale, 0.8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), (60, 60, 230), -1)
        cv2.putText(out, text, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def analyze(image_bgr: np.ndarray, conf: float = 0.2, model_path: str = DEFAULT_MODEL):
    dets = detect(image_bgr, conf, model_path)
    return annotate(image_bgr, dets), severity(dets, image_bgr.shape)


def build_report_html(name: str, original_bgr: np.ndarray, annotated_bgr: np.ndarray, a: Analysis) -> str:
    import base64, datetime, html

    def b64(img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()

    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
                   for k, v in sorted(a.counts.items(), key=lambda kv: -kv[1])) or \
        "<tr><td colspan=2>No waste detected</td></tr>"
    steps = "".join(f"<li>{html.escape(x)}</li>" for x in a.actions)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>WasteDetect report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#222}}
.badge{{display:inline-block;padding:6px 16px;border-radius:20px;color:#fff;font-weight:700;background:{a.color}}}
img{{max-width:100%;border-radius:8px}} .g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}
.score{{font-size:48px;font-weight:800;color:{a.color}}}</style></head><body>
<h1>WasteDetect AI: Litter Report</h1>
<p>Image: <b>{html.escape(name)}</b> &middot; Generated {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
<p class="score">{a.score:.0f}/100 <span class="badge">{a.level} priority</span></p>
<p>{len(a.detections)} items detected &middot; {a.coverage:.0%} of the image covered by detections</p>
<div class="g"><div><h3>Original</h3><img src="data:image/jpeg;base64,{b64(original_bgr)}"></div>
<div><h3>Detections</h3><img src="data:image/jpeg;base64,{b64(annotated_bgr)}"></div></div>
<h3>Items by type</h3><table><tr><th>Type</th><th>Count</th></tr>{rows}</table>
<h3>Recommended next steps</h3><ul>{steps}</ul>
<p style="color:#777;font-size:12px">Score = 60% confidence-weighted item count + 40% image area covered by detections.
AI estimate from a single photo; verify on site.</p></body></html>"""
