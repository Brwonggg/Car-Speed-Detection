import torch
from ultralytics import YOLO

def model_setup():
    """Sets up the object detection model used to identify and track cars driving

    Args: 
        None

    Returns: 
        YOLO model used for object detection
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    
    return model