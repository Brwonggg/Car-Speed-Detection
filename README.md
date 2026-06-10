## Preview(video demonstration)

## Intro 

This is a practice project that utilises computer vision/object detection.

## Technologies
- Python

## Running the Project
To run the project in your own local environment, follow these steps:

1. **Clone the repository to your local machine**
2. Run 'pip install -r requirements.txt'
3. **Drag the traffic video you want to inspect into your terminal**
4. **Copy the file path and paste it into the variable FILE_PATH in main.py**
5. **Run the code**

## Features 

Input the file path of the traffic camera video you want to monitor inside the variable FILE_PATH. A window will then open on your device displaying the same video you have uploaded and will detect and identify cars by bounding them in a green rectangle and also indicating estimates of their speed above it. It will also automatically issue tickets to cars which exceed the value of SPEED_LIMIT in detect_car.py and store the values of detected speed and fine amount in a txt file.


## The process (How I built it)

was intially using a pytorch model but ran into real time object detection limittations and so decided to swap over to ultralytics YOLO model 

the speed is too high because im not tracking the frame rate of the actual video, solved this by implementing frame count and car tracker into the function 

have the issue of the initial calculation appearing too slowly and the subsequent one chaning way too rapidly so i decided to separate them by having initial after 3 frames and the subsequent only update every 20 frames so its more stable, increasing the value of frames_since_update lets the speed text be more stable and consistent as opposed to rapidly changing and found the best results with 60

was initally using backgroundsubtractorMOG2 but YOLO replaced that entirely. Your old code used MOG2 to detect motion before passing to the model. Now YOLO does detection and tracking in one call without needing MOG2 at all.

after it has appeared in at least 20 frames, this is to ensure that the initial speed, which is well above what the car is actually moving at, is not taken 

## What I learnt 

results[0].boxes
 returns a Boxes object containing:
 .xyxy — bounding box coordinates [x1, y1, x2, y2]
tensor([[ 100.,  200.,  300.,  400.],   # box 1 — car
        [ 450.,  150.,  650.,  350.],   # box 2 — car  
        [ 200.,  300.,  280.,  380.]])  # box 3 — person
.cls — class label for each box
tensor([2., 2., 0.])    # 2=car, 2=car, 0=person
 .conf — confidence score for each box
tensor([0.92, 0.87, 0.45])
 .id — tracking ID (only when using model.track)
tensor([3., 7., 12.])   # car3, car7, person12

dont reset your model by putting it inside of a loop or else it will cause choppy video/low fps

## Limitations

While working on this, I found 3 main limitations of this real-time object detection system.

One of which was that it's incapable of being reused across different traffic camera videos without modifications to the code, specifically the PPM and FRAME_RATE. This is because the videos are all different in certain ways, the framing of the angle, the mounted height of the camera, whether it's a fixed or rotating camera etc. As a result, the constants such as the PPM and FRAME_RATE vary drastically and must be refactored when using a different video from the one I used in the example video, if not the speed calculations using estimate_speed() will be very far off the actual value.

I also found difficulty tracking larger vehicles such as trucks and buses because they take up more pixels than cars so I ran into issues of the model not detecting trucks or anything that wasn't a car, it also struggled to calculate the speed of it while it was in the frame. The pixel per metre(PPM) also varies so drastically between the three vehicle classes that estimate_speed() was incapable of giving an accurate assessment even when it did work.

The last limitation I found was that even when only one video was used, a value for the PPM was hard to obtain because it changes constantly throughout the frames. For example, when the car moves away from the camera, it appears smaller and has a higher PPM and conversely, the closer it is, the larger it appears and the smaller the PPM. This once again causes estimate_speed() to calculate inaccurate speeds and possibly incorrectly issue tickets.

## How can it be improved 

For the first limitation, use video footage where the conditions of the camera such as its angle and height are identical to how it would be set up normally. You would then run tests with vehicles moving at known speeds, checking what the model would give for an estimate and then refactor the values of the constant PPM accordingly. For the FRAME_RATE, the frames-per-second of the video can be determined using .get(cv2.CAP_PROP_FPS) from the OpenCV library, then use that value obtained as the value of the constant FRAME_RATE. The frame rate can be set as a variable of the function mentioned instead of being hard-coded.

Another solution for a value of the constant PPM is to use a standard reference from the video, such as a traditional road marking or something which you know the actual length of. You would then get the pixel distance by either inspecting the video in a tool like paint/preview or by using setMouseCallback() from the OpenCV library. You would then take the pixel distance divided by the real life distance to get the value of PPM. Using a traditional road marking acts as a ground truth reference and the pixel inspection gives an accurate value for pixel distance, both of which contribute to a more accurate PPM value. 

For the second limitation of being unable to track trucks and buses due to their larger vehicles sizes, it is possible to set up different vehicle class recognition with the YOLO model. In detect_car.py, the YOLO model rejects any cls output which isn't 2, 2 is the value of the cls used to describe a car. You can then make a list of allowed vehicles classes also containing the values 5 and 7 which represent buses and trucks respectively. However, you would then also need to refactor the PPM values for each vehicle class separately and multiple other things so as to ensure estimate_speed() provides accurate values for all vehicle types.









