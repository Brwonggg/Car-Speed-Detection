from ticket import issue_ticket
from speed import estimate_speed
from model_setup import model_setup
from detect_car import detect_car
import numpy as np
from unittest.mock import patch, MagicMock

def test_issue_ticket(tmp_path):
    test_file = tmp_path / "Speeding_Ticket.txt"
    with patch("ticket.FINE_AMOUNT", 25):
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            issue_ticket(75.5)

            mock_file.write.assert_any_call("Speeding Ticket\n")
            mock_file.write.assert_any_call("Detected Speed: 75.50 km/h\n")

def test_estimate_speed():
    pos1 = [0, 0]
    pos2 = [100, 0]  
    PPM = 10         
    FRAME_RATE = 30
    frames_elapsed = 30         
    speed = estimate_speed(pos1, pos2, PPM, FRAME_RATE, frames_elapsed)
    assert round(speed, 1) == 36.0

def test_model_setup():
    from ultralytics import YOLO
    model = model_setup()
    assert isinstance(model, YOLO)

def test_detect_car():
    frame = np.zeros((480, 640, 3), dtype=np.uint8) 
    car_tracker = {}
    mock_model  = MagicMock()

    #Simulate no detections — boxes.id is None
    mock_model.track.return_value = [MagicMock()]
    mock_model.track.return_value[0].boxes.id = None

    result = detect_car(frame, 31, 10, 1, car_tracker, mock_model)

    #Frame returned unchanged, car_tracker still empty
    assert result is frame
    assert car_tracker == {}