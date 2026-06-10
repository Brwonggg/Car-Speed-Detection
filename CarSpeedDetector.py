import cv2 as cv
import numpy as np
import torch
import torchvision
import math 
import time
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

WEIGHTS = "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT"
model = YOLO("yolov8n.pt")

model = model.to(device)
print(f"Using: {device}")

model.eval()

CONFIDENCE_THRESHOLD = 0.3
FRAME_RATE = 31
PPM = 10
SPEED_LIMIT = 50
FINE_AMOUNT = 25


def estimate_speed(pos1, pos2, PPM, FRAME_RATE):
    dist_pixels = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    dist_meter = dist_pixels / PPM
    speed = dist_meter / FRAME_RATE * 3.6
    return speed 

def detect_car(frame, FRAME_RATE, PPM):
    global pos_list_prev
    results = model(frame, verbose=False)
    
    for i, box in enumerate(results[0].boxes):
        cls   = int(box.cls[0])
        score = float(box.conf[0])

        if cls != 2 or score < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if score >= CONFIDENCE_THRESHOLD:
            cv.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 2)
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            pos_list = [centroid_x, centroid_y]

            if pos_list_prev is not None:
                speed = estimate_speed(pos_list_prev, pos_list, PPM, FRAME_RATE)
                speed_text = int(speed)
                if speed >= SPEED_LIMIT:
                    issue_ticket(speed)
                    print("Speeding ticket issued")
            else:
                speed = 0
                speed_text = "calculating..."
            
            cv.putText(frame, f"Speed: {speed_text}", (x1,y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),2)
            pos_list_prev = pos_list
    return frame
    
def issue_ticket(speed):
    filename = "Speeding_Ticket.txt"
    with open (filename,"a") as file:
        file.write("Speeding Ticket\n")
        file.write(f"Detected Speed: {speed:.2f} km/h\n")
        file.write(f"Fine Amount: ${FINE_AMOUNT}\n")
        file.write("\n")
    
frame_count = 0

cap = cv.VideoCapture('/Users/brandon/Downloads/NEW Moving Cars.mov')
if not cap.isOpened():
    print("Error: Unable to load video")
    exit()
pos_list_prev = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # skip every other frame — halves model calls
    if frame_count % 2 != 0:
        continue

    # resize to 25% — 2880x1864 → 720x466
    #frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    start = time.time()
    processed_frame = detect_car(frame, FRAME_RATE, PPM)
    end = time.time()
    print(f"Inference: {(end-start)*1000:.1f}ms")
    
    cv.imshow("Cars Driving",processed_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()