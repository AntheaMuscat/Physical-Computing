import time
import pigpio
import subprocess
import os

# Pin configuration
servo_pin = 26  # GPIO26 (physical pin 37)

# Function to start pigpiod
def start_pigpiod():
    """
    Starts the pigpiod daemon if not already running.
    """
    try:
        subprocess.run(["sudo", "pigpiod"], check=True)
        time.sleep(1)  # Give the daemon time to start
    except subprocess.CalledProcessError as e:
        print(f"Error starting pigpiod: {e}")
        exit(1)

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

# Main script
if __name__ == "__main__":
    # Start the pigpiod daemon
    if not os.system("pgrep pigpiod"):
        print("pigpiod already running.")
    else:
        print("Starting pigpiod daemon...")
        start_pigpiod()

    # Initialize pigpio
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon!")
        exit(1)

    try:
        while True:
            set_angle(90)  # Rotate to 90 degrees
            print("Servo at 90 degrees")
            time.sleep(5)  # Wait for 5 seconds
            set_angle(0)   # Rotate to 0 degrees
            print("Servo at 0 degrees")
            time.sleep(5)  # Wait for 5 seconds
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup
        pi.set_servo_pulsewidth(servo_pin, 0)  # Stop sending PWM signals
        pi.stop()
        print("Servo stopped.")