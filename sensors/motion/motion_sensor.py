from gpiozero import MotionSensor
from signal import pause

pir = MotionSensor(4)

while True:
    pir.wait_for_motion()
    print("Motion Detected")
    pir.wait_for_no_motion()
    print("Motion Stopped")