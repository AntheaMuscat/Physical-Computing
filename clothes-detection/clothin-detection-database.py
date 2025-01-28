import tkinter as tk
from tkinter import PhotoImage, Button, Label, ttk
from PIL import Image, ImageTk
from config import insert_into_database, fetch_data, remove_from_database
import cv2
import subprocess
import os
import time
import Adafruit_DHT

import board
import neopixel
from datetime import datetime

# Import necessary components
from gpiozero import MotionSensor
import gpiozero as gpio
from threading import Thread

from gpiozero import Button as GPIOButton

# Motion Sensor
pir = MotionSensor(4)
motionDetectedCounter = 0
noMotionDetectedCounter = 0

# Servo Initialization with pigpio
import pigpio

servo_pin = 26
pi = pigpio.pi()

push_button = GPIOButton(21)


if not pi.connected:
    print("Failed to connect to pigpio daemon!")
    exit(1)

# Humidity Sensor
DHTPin = 17
sensor = Adafruit_DHT.DHT11

# NeoPixel LED Setup
PIXEL_PIN = board.D18
NUM_PIXELS = 43
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=1.0, auto_write=True)

class SimpleCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Camera App")

        # Set up video capture
        self.cap = cv2.VideoCapture(0)
        self.led_brightness = 1

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Error: Camera could not be initialized.")
            self.cap.release()
            return

    
        # Create the main container frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top part: Left side for LEDs and Right side for camera with humidity label
        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X,expand=True, padx=500, pady=10)

        



        # Left frame for LEDs buttons and LED mode label
        left_frame = tk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # LED mode label (Auto updates based on time)
        self.led_mode_label = Label(left_frame, font=("Arial", 14),  width=15, anchor="w")
        self.led_mode_label.pack(pady=5)

         # Day/Night Toggle Buttons
        self.day_mode_button = Button(root, text="Day Mode", command=self.set_day_mode)
        self.day_mode_button.pack(side=tk.LEFT, padx=10)

        self.night_mode_button = Button(root, text="Night Mode", command=self.set_night_mode)
        self.night_mode_button.pack(side=tk.LEFT, padx=10)

        


        # Buttons to change LED colors
        self.day_mode_button = Button(left_frame, text="Day Mode", command=self.set_day_mode, bg="white")
        self.day_mode_button.pack(pady=5)

        self.night_mode_button = Button(left_frame, text="Night Mode", command=self.set_night_mode, bg="#C49327", fg="white")
        self.night_mode_button.pack(pady=5)

        self.red_button = Button(left_frame, text="Red", command=self.set_red, bg="red", fg="white")
        self.red_button.pack(pady=5)

        self.blue_button = Button(left_frame, text="Blue", command=self.set_blue, bg="blue", fg="white")
        self.blue_button.pack(pady=5)

        self.green_button = Button(left_frame, text="Green", command=self.set_green, bg="green",fg="white")
        self.green_button.pack(pady=5)

        self.yellow_button = Button(left_frame, text="Yellow", command=self.set_yellow, bg="yellow")
        self.yellow_button.pack(pady=5)

        self.pink_button = Button(left_frame, text="Pink", command=self.set_pink, bg="pink")
        self.pink_button.pack(pady=5)

        self.purple_button = Button(left_frame, text="Purple", command=self.set_purple, bg="purple", fg="white")
        self.purple_button.pack(pady=5)

        # Add brightness control slider
        self.brightness_label = Label(left_frame, text="Brightness", font=("Arial", 12))
        self.brightness_label.pack(pady=5)

        style = ttk.Style()

        # Configure the style for the scale (slider)
        style.configure("TScale",
                sliderlength=20,      # Length of the slider
                thickness=12,         # Thickness of the scale
                background="#3498db", # Slider color
                troughcolor="#ecf0f1", # Track color
                foreground = "#113038"
               )

        style.map("TScale",
          background=[("active", "#3498db")]      # Set hover foreground color (handle)
)


        self.brightness_slider = ttk.Scale(left_frame, from_=0, to=1, orient="horizontal", command=self.adjust_brightness,)
        self.brightness_slider.set(self.led_brightness)  # Set the default brightness to 1
        self.brightness_slider.pack(pady=5)

        # Right frame for camera feed and humidity label
        right_frame = tk.Frame(top_frame)
        right_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Camera feed
        self.img_label = tk.Label(right_frame)
        self.img_label.pack()

        # Humidity label below camera feed
        self.humidity_label = Label(right_frame, text="Humidity: -", font=("Arial", 14))
        self.humidity_label.pack(pady=5)


        # Link the physical button to the start_capture_timer method
        push_button.when_pressed = self.start_capture_timer
 

        # Start video feed update
        self.update_video_feed()
        self.update_humidity_display()
        self.update_led_mode()

        # Bottom part: Treeview
        table_frame = tk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.table = ttk.Treeview(
            table_frame,
            columns=("Index", "Clothing Item", "Colour", "Date Added"),
            show="tree headings",
            yscrollcommand=v_scrollbar.set,
        )

        # Configure the first column (#0) for images
        self.table.column("#0", width=120, anchor="center")  # Adjust width for the image
        self.table.heading("#0", text="Image")  # Set heading for the image column

        # Configure other columns
        self.table.column("Index", width=80, anchor="center")
        self.table.column("Clothing Item", width=150, anchor="w")
        self.table.column("Colour", width=100, anchor="center")
        self.table.column("Date Added", width=100, anchor="center")

        # Set headings for other columns
        self.table.heading("Index", text="Index")
        self.table.heading("Clothing Item", text="Clothing Item")
        self.table.heading("Colour", text="Colour")
        self.table.heading("Date Added", text="Date Added")

        self.table.pack(fill=tk.BOTH, expand=True)
        ttk.Style().configure("Treeview", rowheight=200)

        v_scrollbar.config(command=self.table.yview)

        # Populate the table
        self.update_data_table()

    def adjust_brightness(self, value):
        self.led_brightness = float(value)
        pixels.brightness = self.led_brightness  # Update the NeoPixel brightness
        pixels.show()  # Apply the brightness adjustment

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
        else:
            print("Error: Unable to capture frame.")        
        self.root.after(10, self.update_video_feed)  # Refresh video feed


    def start_capture_timer(self):
        self.add_button.config(state=tk.DISABLED)
        self.countdown(5)

    def countdown(self, remaining_time):
        if remaining_time > 0:
            self.root.after(1000, self.countdown, remaining_time - 1)
        else:
            self.capture_and_save_image()
            self.add_button.config(state=tk.NORMAL)

    def capture_and_save_image(self):
        ret, frame = self.cap.read()
        if ret:
            filename = f"captured_image.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            subprocess.run(["python3", "clothing_detection.py", "-d", "df2"])
            self.update_data_table()
        else:
            print("Error: Failed to capture image.")

    def update_data_table(self):
        data = fetch_data()

        # Clear the existing rows in the table
        for row in self.table.get_children():
            self.table.delete(row)

        # Dictionary to store PhotoImage objects to prevent garbage collection
        self.table.image_dict = {}

        # Process each row
        for idx, row in enumerate(data):
            image_path = row[4]  # Assuming the image path is in the 5th column
            img_tk = None

            if os.path.exists(image_path):
                # Open and resize the image
                img = Image.open(image_path).resize((200, 200), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.table.image_dict[idx] = img_tk  # Prevent garbage collection

            # Insert the row, placing the image in the first column
            self.table.insert(
                "", tk.END, text="", image=img_tk, values=row[:4]
            )

    def remove_selected_item(self):
        selected_item = self.table.selection()
        if selected_item:
            item_index = self.table.item(selected_item[0])['values'][0]
            remove_from_database(item_index)
            self.update_data_table()
            self.table.update_idletasks()
        else:
            print("No item selected to remove.")

    def update_led_mode(self):
        time_now = datetime.now().hour
        if 8 <= time_now < 17:  # Daytime
            self.led_mode_label.config(text="LED Mode: Day")
        else:  # Nighttime
            self.led_mode_label.config(text="LED Mode: Night")


    # Button commands for changing LED colors
    def set_day_mode(self):
        self.led_mode_label.config(text="LED Mode: Day")
        pixels.fill((255, 255, 255))  # Bright White
        pixels.show()

    def set_night_mode(self):
        self.led_mode_label.config(text="LED Mode: Night")
        pixels.fill((196, 147, 39))  # Warm White
        pixels.show()

    def set_red(self):
        self.led_mode_label.config(text="LED Mode: Red")
        pixels.fill((255, 0, 0))  # Red
        pixels.show()

    def set_blue(self):
        self.led_mode_label.config(text="LED Mode: Blue")
        pixels.fill((0, 0, 255))  # Blue
        pixels.show()

    def set_green(self):
        self.led_mode_label.config(text="LED Mode: Green")
        pixels.fill((0, 255, 0))  # Green
        pixels.show()

    def set_yellow(self):
        self.led_mode_label.config(text="LED Mode: Yellow")
        pixels.fill((255, 255, 0))  # Yellow
        pixels.show()

    def set_pink(self):
        self.led_mode_label.config(text="LED Mode: Pink")
        pixels.fill((255, 20, 147))  # Pink
        pixels.show()

    def set_purple(self):
        self.led_mode_label.config(text="LED Mode: Purple")
        pixels.fill((128, 0, 128))  # Purple
        pixels.show()

    def __del__(self):
        self.cap.release()  # Release the video capture when app is closed

    
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SimpleCameraApp(root)
#     root.mainloop()


    def update_humidity_display(self):
        humidity, temperature = Adafruit_DHT.read_retry(sensor, DHTPin)
        if humidity is not None:
            # humidity = round(humidity)
            # humidity_str = str(humidity)
            self.humidity_label.config(text=f"Humidity: {humidity:.2f}%")

            if humidity >= 60:
                self.humidity_label.config(text=f"Humidity: {humidity:.2f}%\n High Humidity please check clothes", fg="red")
        else:
            self.humidity_label.config(text="Failed to retrieve humidity data.")

        time.sleep(5)
        self.root.after(600000, self.update_humidity_display)  # Update every 2 seconds



class MotionHandler:
    def __init__(self):
        self.motion_count = 0
        self.running = True  # Control flag to stop the loops

        # Initialize the LED strip
        self.pixels1 = pixels  # NeoPixel object for motion handling

        # Constants for time ranges
        self.DAY_START = 8  # 8:00 AM
        self.DAY_END = 17   # 5:00 PM

    def start_motion_sensor(self):
        global motionDetectedCounter, noMotionDetectedCounter
        while True:
            if pir.motion_detected:
                motionDetectedCounter += 1
                noMotionDetectedCounter = 0
                print("Motion Detected: " + str(motionDetectedCounter))

                if motionDetectedCounter >= 2: 
                    self.open_camera_app()
            else:
                noMotionDetectedCounter +=1
                motionDetectedCounter = 0
                print("No Motion Detected")
            time.sleep(1)

    def start_servo(self):
        global pi, servo_pin
        while self.running:
            pi.set_servo_pulsewidth(servo_pin, 500)  # Move to 0 degrees
            time.sleep(1)  # Hold position for a while
            pi.set_servo_pulsewidth(servo_pin, 2450)  # Move to 180 degrees
            time.sleep(1)
            pi.set_servo_pulsewidth(servo_pin, 500)  # Move to 0 degrees
            time.sleep(1) 

            time.sleep(5)  # Wait for 5 minutes before pressing again

    def start_humidity_monitor(self):
        import Adafruit_DHT
        while self.running:
            humidity, temperature = Adafruit_DHT.read_retry(sensor, DHTPin)
            if humidity is not None:
                print(f"Humidity: {humidity:.2f}%")
                print(f"Temperature: {temperature:.2f}°C")
            else:
                print("Failed to retrieve data from humidity sensor.")
            time.sleep(2)

    def start_leds(self):
        

        time_now = datetime.now().hour
        if self.DAY_START <= time_now < self.DAY_END:  # Daytime
            self.pixels1.fill((255, 255, 255))  # Bright White
        else:  # Nighttime
            self.pixels1.fill((196,147,39))  # Warm White

    def open_camera_app(self):
        global root
        root = tk.Tk()
        app = SimpleCameraApp(root)

        # Start servo and humidity monitor in separate threads
        Thread(target=self.start_servo).start()
        # Thread(target=self.start_humidity_monitor).start()
        Thread(target=self.start_leds).start()
        
        root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        root.mainloop()

    def on_app_close(self): 
        global root, pi 
        self.running = False # Set the running flag to False 
        # Safely destroy the root window in the main thread 
        self.pixels1.fill((0,0,0))
        root.after(0, root.destroy) # Ensure this is in the main thread 
        pi.stop() # Stop the pigpio connection
        quit()

if __name__ == "__main__":
    try:
        motion_handler = MotionHandler()
        motion_thread = Thread(target=motion_handler.start_motion_sensor)
        motion_handler.pixels1.fill((0,0,0))
        motion_thread.start()
        

    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        # Perform cleanup actions if necessary
        motion_handler.on_app_close()
        quit()


