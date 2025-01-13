import time
import board
import neopixel
from datetime import datetime

# Initialize the LED strip
pixels1 = neopixel.NeoPixel(board.D18, 43, brightness=1)

# Constants for time ranges
DAY_START = 8  # 8:00 AM
DAY_END = 17   # 5:00 PM

# Function to determine if it's daytime or nighttime
def check_time():
    time_now = datetime.now().hour
    if DAY_START <= time_now < DAY_END:  # Daytime
        pixels1.fill((255, 255, 255))  # Bright White
    else:  # Nighttime
        pixels1.fill((196,147,39))  # Warm White

# Main loop to dynamically update LED colors
try:
    while True:
        check_time()
        time.sleep(60)  # Check every 60 seconds
except KeyboardInterrupt:
    # Turn off LEDs gracefully on exit
    pixels1.fill((0, 0, 0))
