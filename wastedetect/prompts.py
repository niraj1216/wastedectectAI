"""Litter vocabulary shared by the exporter and the runtime.
Order matters: it defines the class index of the exported ONNX model.
text prompt -> (report label, severity weight)"""
WORLD_PROMPTS = {
    "garbage pile": ("garbage pile", 3.0),
    "trash bag": ("trash bag", 1.5),
    "plastic bag": ("plastic bag", 0.8),
    "plastic bottle": ("plastic bottle", 1.0),
    "aluminum can": ("can", 0.8),
    "cardboard box": ("cardboard", 1.2),
    "paper litter": ("paper litter", 0.6),
    "food wrapper": ("wrapper", 0.6),
    "disposable cup": ("cup", 0.8),
    "overflowing trash bin": ("overflowing bin", 2.5),
    "discarded furniture": ("bulky waste", 2.5),
    "tire": ("tire", 1.5),
    "debris": ("debris", 1.0),
}
