from gpiozero import Button
import time

button = Button(21)  # Ensure the correct GPIO pin

def on_press():
    print("Button Pressed!")

button.when_pressed = on_press

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
