import time
import board
import neopixel
from datetime import datetime
 
  # Initialize the LED strip
pixels1 = neopixel.NeoPixel(board.D18, 55, brightness=1)
  
 # Function to get the current hour
def current_hour():
     return datetime.now().hour
 
 # Function to determine if it's daytime or nighttime
def is_daytime():
     return current_hour() < 17  # 5 PM in 24-hour format
  
 # Set the initial color based on the time of day
if is_daytime():
     pixels1.fill((255, 255, 255))  # White
     time.sleep(5)
else:
     pixels1.fill((255, 240, 200))  # Warm White
     time.sleep(5)
