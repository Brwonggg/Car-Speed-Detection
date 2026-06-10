import cv2 as cv
import time
from detect_car import detect_car
from model_setup import model_setup

FRAME_RATE = 31
PPM = 10
FILE_PATH = "/Users/brandon/Downloads/NEW Moving Cars.mov"

frame_count = 0
car_tracker = {}

cap = cv.VideoCapture(FILE_PATH)
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
    
    model = model_setup()

    start = time.time()
    processed_frame = detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker, model)
    end = time.time()
    print(f"Inference: {(end-start)*1000:.1f}ms")
    
    cv.imshow("Cars Driving",processed_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()