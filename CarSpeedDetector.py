import cv2 as cv
import numpy as np
import torch
import torchvision
import math 
import time

WEIGHTS = "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT"
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=WEIGHTS)
model.eval()

CONFIDENCE_THRESHOLD = 0.5
FRAME_RATE = 33
PPM = 8.8
SPEED_LIMIT = 50
FINE_AMOUNT = 25

def estimate_speed(pos1, pos2, PPM, FRAME_RATE, frames_elapsed):
    dist_pixels = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    dist_meter = dist_pixels / PPM
    time_elapsed = frames_elapsed / FRAME_RATE
    speed = dist_meter / time_elapsed * 3.6
    return speed 

def detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker):
    global pos_list_prev
    img_tensor = torchvision.transforms.ToTensor()(frame)
    with torch.inference_mode():
        preds = model(img_tensor.unsqueeze(dim=0))
        for i, (pred, score) in enumerate(zip(preds[0]["boxes"], preds[0]["scores"])):
            x1, y1, x2, y2 = map(int, pred)
            if score >= CONFIDENCE_THRESHOLD:
                cv.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 2)
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                pos_list = [centroid_x, centroid_y]

                if i not in car_tracker:
                    car_tracker[i] = {"pos": pos_list, "frame": frame_count,"speed": None}
                    speed_text = "calculating..."

                else:
                    pos1 = car_tracker[i]["pos"]
                    frame_started  = car_tracker[i]["frame"]
                    frames_elapsed = frame_count - frame_started

                    if frames_elapsed >= 3:   # wait at least 10 frames for stable reading
                        speed = estimate_speed(pos1, pos_list, PPM, FRAME_RATE, frames_elapsed)
                        car_tracker[i]["speed"] = speed
                        speed_text = f"{speed:.1f} km/h"

                        
                    else:
                        speed_text = "calculating..."
                        
                    # if pos_list_prev is not None:
                    #     speed = estimate_speed(pos_list_prev, pos_list, PPM, FRAME_RATE, frames_elapsed)
                    #     if speed >= SPEED_LIMIT:
                    #         issue_ticket(speed)
                    #         print("Speeding ticket issued")
                    car_tracker[i]["pos"] = pos_list
                    car_tracker[i]["frame"] = frame_count
                    
                stored_speed = car_tracker[i].get("speed")
                if stored_speed is not None:
                    speed = estimate_speed(pos_list_prev, pos_list, PPM, FRAME_RATE, frames_elapsed)
                    if speed >= SPEED_LIMIT:
                        issue_ticket(speed)
                        print("Speeding ticket issued")
                    speed_text = f"{stored_speed:.1f}"
                else:
                    speed = 0
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
car_tracker = {}

cap = cv.VideoCapture("/Users/brandon/Downloads/Cars Driving NEW.mov")
if not cap.isOpened():
    print("Error: Unable to load video")
    exit()
pos_list_prev = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    processed_frame = detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker)

    
    cv.imshow("Cars Driving", processed_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()