import cv2 as cv
from detect_car import detect_car
from model_setup import model_setup

FRAME_RATE = 31
PPM = 10
FILE_PATH = '/Users/brandon/Downloads/NEW Moving Cars.mov'

frame_count = 0
car_tracker = {}

model = model_setup()

cap = cv.VideoCapture(FILE_PATH)
if not cap.isOpened():
    print("Error: Unable to load video")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_count += 1

    #Skip every other frame — fix low fps
    if frame_count % 2 != 0:
        continue

    processed_frame = detect_car(frame, FRAME_RATE, PPM, frame_count, car_tracker, model)
    
    cv.imshow("Cars Driving",processed_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()