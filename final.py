from gpiozero import MotionSensor
import time
import subprocess
import pigpio
from datetime import datetime
import picamera

# Pin configuration
motion_sensor_pin = 4
servo_pin = 26  # GPIO26 (physical pin 37)
camera = picamera.PICamera()  # Camera initialization

# Initialize pigpio
pi = pigpio.pi()
if not pi.connected:
    print("Failed to connect to pigpio daemon!")
    exit(1)

# Start pigpiod if not running
if not subprocess.run(["pgrep", "pigpiod"]).returncode == 0:
    subprocess.run(["sudo", "pigpiod"])

# Motion sensor setup
pir = MotionSensor(motion_sensor_pin)

# Servo parameters
MIN_PULSE_WIDTH = 500
MAX_PULSE_WIDTH = 2500

def set_angle(angle):
    """
    Set the servo angle.
    Angle range: 0 to 180 degrees.
    """
    pulse_width = MIN_PULSE_WIDTH + (angle / 180.0) * (MAX_PULSE_WIDTH - MIN_PULSE_WIDTH)
    pi.set_servo_pulsewidth(servo_pin, pulse_width)

def capture_image():
    """
    Capture an image with the camera.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"motion_{timestamp}.jpg"
    camera.capture(image_path)
    print(f"Image captured: {image_path}")

def loop():
    count = 0
    while True:
        pir.wait_for_motion()
        count += 1
        print("Motion Detected", count)
        if count >= 3:
            print("Turning on camera, servo, and humidity sensor...")
            capture_image()
            set_angle(90)  # Start servo at 90 degrees
            # Add your humidity sensor reading code here
            # Example: 
            # humidity = read_humidity()
            # print(f"Humidity: {humidity}")
            count = 0  # Reset motion count
        pir.wait_for_no_motion()
        print("Motion Stopped")

if __name__ == '__main__':
    try:
        loop()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup
        pi.set_servo_pulsewidth(servo_pin, 0)  # Stop sending PWM signals
        pi.stop()
        camera.close()
