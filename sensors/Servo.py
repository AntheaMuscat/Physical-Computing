import time
import pigpio

# Pin configuration
servo_pin = 18  # GPIO17 (physical pin 11)

# Initialize pigpio library
pi = pigpio.pi()

if not pi.connected:
    print("Failed to connect to pigpio daemon!")
    exit()

# Servo parameters
# Pulse width in microseconds: 500 (0°), 1500 (90°), 2500 (180°)
MIN_PULSE_WIDTH = 500
MAX_PULSE_WIDTH = 2500

def set_angle(angle):
    """
    Set the servo angle.
    Angle range: 0 to 180 degrees.
    """
    pulse_width = MIN_PULSE_WIDTH + (angle / 180.0) * (MAX_PULSE_WIDTH - MIN_PULSE_WIDTH)
    pi.set_servo_pulsewidth(servo_pin, pulse_width)

try:
    while True:
        set_angle(90)  # Rotate to 90 degrees
        time.sleep(5)  # Wait for 5 seconds
        set_angle(0)   # Rotate back to 0 degrees
        time.sleep(5)  # Wait for 5 seconds
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    pi.set_servo_pulsewidth(servo_pin, 0)  # Stop sending PWM signals
    pi.stop()
