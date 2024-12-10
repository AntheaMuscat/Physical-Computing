import pigpio
import time

# Initialize pigpio
pi = pigpio.pi()

# Check if pigpio is connected
if not pi.connected:
    print("Failed to connect to pigpio daemon")
    exit()

# Define GPIO pins
pir_pin = 4  # PIR sensor pin
led_pin = 17  # LED pin

# Set up GPIO pins
pi.set_mode(pir_pin, pigpio.INPUT)  # PIR as input
pi.set_mode(led_pin, pigpio.OUTPUT)  # LED as output

# Initialize LED state
pi.write(led_pin, 0)  # LED off

try:
    while True:
        # Wait for motion detection
        motion_detected = pi.wait_for_edge(pir_pin, pigpio.RISING_EDGE)  # Wait for motion
        if motion_detected:
            print("Motion Detected")
            pi.write(led_pin, 1)  # Turn on LED
        
        # Wait for no motion detection
        motion_stopped = pi.wait_for_edge(pir_pin, pigpio.FALLING_EDGE)  # Wait for no motion
        if motion_stopped:
            print("Motion Stopped")
            pi.write(led_pin, 0)  # Turn off LED
        
        # Small delay to avoid excessive checking
        time.sleep(0.1)

except KeyboardInterrupt:
    # Gracefully stop the script on Ctrl+C
    print("Exiting program...")

finally:
    # Clean up GPIO and stop pigpio
    pi.write(led_pin, 0)  # Ensure LED is off
    pi.stop()  # Stop the pigpio daemon
