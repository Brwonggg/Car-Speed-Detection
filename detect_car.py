import cv2 as cv
from ticket import issue_ticket
from speed import estimate_speed

CONFIDENCE_THRESHOLD = 0.7
SPEED_LIMIT = 90

frame_count = 0
car_tracker = {}

def detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker, model):
    
    #model.track recognises the same car throughout the frames
    results = model.track(frame, persist=True, verbose=False)
    
    #No detections this frame
    if results[0].boxes.id is None:   
        return frame

    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        cls = int(box.cls[0])
        score = float(box.conf[0])

        if cls != 2 or score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx  = (x1 + x2) / 2
        cy  = (y1 + y2) / 2
        pos = [cx, cy]

        if track_id not in car_tracker:
            car_tracker[track_id] = {
                "pos": pos,
                "frame": frame_count,
                "speed": None,
                "last_update": frame_count,
                "first_seen": frame_count,   
                "readings": 0, 
                "stabilised": False
            }

        else:
            frames_elapsed = frame_count - car_tracker[track_id]["frame"]
            frames_since_update = frame_count - car_tracker[track_id]["last_update"]
            frames_since_first  = frame_count - car_tracker[track_id]["first_seen"]

            #First reading after 2 frames, update every 5 frames after
            if frames_elapsed >= 2 and (car_tracker[track_id]["speed"] is None or frames_since_update >= 5):
                pos1  = car_tracker[track_id]["pos"]
                
                speed = estimate_speed(pos1, pos, PPM, FRAME_RATE, frames_elapsed)
                
                car_tracker[track_id]["speed"] = speed
                car_tracker[track_id]["last_update"] = frame_count
                car_tracker[track_id]["pos"] = pos
                car_tracker[track_id]["frame"] = frame_count
                car_tracker[track_id]["readings"] += 1

                #Ensures speed readings are only registered and shown after stabilising
                if frames_since_first >= 20 and car_tracker[track_id]["readings"] >= 10:
                    car_tracker[track_id]["stabilised"] = True

                    if speed >= SPEED_LIMIT and frames_since_first >= 20 and car_tracker[track_id]["readings"] >= 3:      
                        issue_ticket(speed)
                        print(f"Speeding ticket: {speed:.1f} km/h — car {track_id}")

        stabilised = car_tracker[track_id].get("stabilised", False)
        stored_speed = car_tracker[track_id].get("speed")
        
        if stabilised and stored_speed is not None:
            speed_text = f"{stored_speed:.1f} km/h"
        else:
            speed_text = "..."       
            
        cv.putText(frame, f"Speed: {speed_text}", (x1,y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),2)
        
    return frame

