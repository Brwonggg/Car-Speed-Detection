import math

def estimate_speed(pos1, pos2, PPM, FRAME_RATE, frames_elapsed):
    dist_pixels = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    dist_meter = dist_pixels / PPM
    time_elapsed = frames_elapsed / FRAME_RATE
    speed = dist_meter / time_elapsed * 3.6
    return speed 