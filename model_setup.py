import torch
from ultralytics import YOLO

def model_setup():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = YOLO("yolov8n.pt")

    model = model.to(device)
    print(f"Using: {device}")

    model.eval()

    return model