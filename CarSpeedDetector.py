import cv2 as cv
import torch
import math, time
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model = model.to(device)
print(f"Using: {device}")

model.eval()

CONFIDENCE_THRESHOLD = 0.7
FRAME_RATE = 31
PPM = 10
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
    results = model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes.id is None:   # no detections this frame
        return frame

    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        cls      = int(box.cls[0])
        score    = float(box.conf[0])

        if cls != 2 or score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx  = (x1 + x2) / 2
        cy  = (y1 + y2) / 2
        pos = [cx, cy]

        if track_id not in car_tracker:
            car_tracker[track_id] = {
                "pos":         pos,
                "frame":       frame_count,
                "speed":       None,
                "last_update": frame_count,
                "first_seen":  frame_count,   
                "readings":    0, 
                "stabilised": False
            }

        else:
            frames_elapsed      = frame_count - car_tracker[track_id]["frame"]
            frames_since_update = frame_count - car_tracker[track_id]["last_update"]
            frames_since_first  = frame_count - car_tracker[track_id]["first_seen"]

            # first reading after 2 frames, update every 10 frames after
            if frames_elapsed >= 2 and (car_tracker[track_id]["speed"] is None or frames_since_update >= 5):
                pos1  = car_tracker[track_id]["pos"]
                speed = estimate_speed(pos1, pos, PPM, FRAME_RATE, frames_elapsed)
                car_tracker[track_id]["speed"]       = speed
                car_tracker[track_id]["last_update"] = frame_count
                car_tracker[track_id]["pos"]         = pos
                car_tracker[track_id]["frame"]       = frame_count
                car_tracker[track_id]["readings"]   += 1

                if frames_since_first >= 20 and car_tracker[track_id]["readings"] >= 10:
                    car_tracker[track_id]["stabilised"] = True

                if speed >= SPEED_LIMIT and frames_since_first >= 20 and car_tracker[track_id]["readings"] >= 3:      
                    issue_ticket(speed)
                    print(f"Speeding ticket: {speed:.1f} km/h — car {track_id}")

        stabilised   = car_tracker[track_id].get("stabilised", False)
        stored_speed = car_tracker[track_id].get("speed")
        if stabilised and stored_speed is not None:
            speed_text = f"{stored_speed:.1f} km/h"
        else:
            speed_text = "..."       
            
        cv.putText(frame, f"Speed: {speed_text}", (x1,y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),2)
        
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

cap = cv.VideoCapture('/Users/brandon/Downloads/Cars Driving NEW.mov')
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
    processed_frame = detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker)
    end = time.time()
    print(f"Inference: {(end-start)*1000:.1f}ms")
    
    cv.imshow("Cars Driving",processed_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()