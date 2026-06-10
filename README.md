video returns two dimensions and doesnt have the colour channels so i had to unsqueeze it in order to match 

the speed is too high because im not tracking the frame rate of the actual video, solved this by implementing frame count and car tracker into the function 

have the issue of the initial calculation appearing too slowly and the subsequent one chaning way too rapidly so i decided to separate them by having initial after 3 frames and the subsequent only update every 20 frames so its more stable, increasing the value of frames_since_update lets the speed text be more stable and consistent as opposed to rapidly changing and found the best results with 60

another limitation is the ppm which is hard to obtain because the further away it moves, it appears smaller and has a higher ppm and the closer it is, the larger it appears and smaller the pixel per metre so try to use a standard reference from the video like a traditional road marking to have a more accurate gauge

higher resolution videos will appear much more choppy and lower fps so resize the frame

was intiially using a pytorch model but ran into real time object detection limittations and so decided to swap over to ultralytics YOLO model 

was initally using backgroundsubtractorMOG2 but YOLO replaced that entirely. Your old code used MOG2 to detect motion before passing to the model. Now YOLO does detection and tracking in one call without needing MOG2 at all.

