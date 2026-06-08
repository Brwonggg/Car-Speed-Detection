import cv2 as cv
import numpy as np
import torch
import torchvision
import math 
import time

WEIGHTS = "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT"
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=WEIGHTS)
model.eval()

confidence_threshold = 0.5
frame_rate = 10
ppm = 8.8

def estimate_speed(pos1, pos2, ppm, frame_rate):
    math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)