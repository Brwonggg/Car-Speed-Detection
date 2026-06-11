import math

def estimate_speed(pos1: list, pos2: list, PPM: int, FRAME_RATE: int, frames_elapsed: int):
    """Calculates an estimate of the speed of a vehicle
    
    Args:
        pos1: list 2 values, intial x and y coordinates
        pos2: list storing 2 values, final x and y coordinates
        PPM: int constant value hard-coded in main.py
        FRAME_RATE: int constant value hard-coded in main.py
        frames_elapsed: int measuring frames since a car was first seen or last updated 

    Returns:
        A float that denotes the speed of a vehicle
    """
    dist_pixels = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    dist_meter = dist_pixels / PPM
    time_elapsed = frames_elapsed / FRAME_RATE
    speed = dist_meter / time_elapsed * 3.6
    return speed 