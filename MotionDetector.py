import cv2 as cv
import numpy as np
import torch 

video_path = '/Users/brandon/Downloads/Cars Driving NEW.mov'
cap = cv.VideoCapture(video_path)
#detects and isolates moving objects in video
backSub = cv.createBackgroundSubtractorMOG2() 

while True:
    #Capture frame by frame, .read() returns a tuple of (boolean, matrix)
    ret, frame = cap.read()
    if not ret:
        break
    fgMask = backSub.apply(frame)
    #RETR_EXTERNAL retrieves only outermost boundaries
    #CHAIN_APPROX_SIMPLE compresses all lines and retains only end points
    #contours,_ is used because findContours() returns two values 
    contours,_ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        min_area = 1000
        if cv.contourArea(contour) > min_area:
            cv.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2)

    cv.imshow("Cars Driving", frame)
    #Quits when user presses key "Q"
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()