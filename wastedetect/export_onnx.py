"""One-off export: YOLO-World (zero-shot) -> fixed-vocabulary ONNX. Needs torch + ultralytics.
Run in Docker build, Colab, or locally:  python export_onnx.py   (EXPORT_IMGSZ=512 for less RAM)"""
import os, shutil
from ultralytics import YOLOWorld
from prompts import WORLD_PROMPTS

imgsz = int(os.getenv("EXPORT_IMGSZ", "640"))
m = YOLOWorld("yolov8s-worldv2.pt")
m.set_classes(list(WORLD_PROMPTS))
out = m.export(format="onnx", imgsz=imgsz, simplify=True, opset=12, dynamic=False)
os.makedirs("models", exist_ok=True)
shutil.copy(out, "models/waste_world.onnx")
print("saved models/waste_world.onnx", os.path.getsize("models/waste_world.onnx") // 1_000_000, "MB")
