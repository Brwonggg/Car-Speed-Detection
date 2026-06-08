import cv2 as cv
import numpy as np
import torch 

video_path = '/Users/brandon/Downloads/Cars Driving NEW.mov'
cap = cv.VideoCapture(video_path)

while True:
    #Capture frame by frame
    ret, frame = cap.read()

    cv.imshow("Cars Driving", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()