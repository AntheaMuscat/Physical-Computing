import tkinter as tk
from tkinter import PhotoImage, Button, Label, ttk
from PIL import Image, ImageTk
from config import insert_into_database, fetch_data, remove_from_database
import cv2
import subprocess
import os
import time
import Adafruit_DHT

# Import necessary components
from gpiozero import MotionSensor
import gpiozero as gpio
from threading import Thread

# Motion Sensor
pir = MotionSensor(4)
motionDetectedCounter = 0
noMotionDetectedCounter = 0

# Servo Initialization with pigpio
import pigpio

servo_pin = 26
pi = pigpio.pi()


if not pi.connected:
    print("Failed to connect to pigpio daemon!")
    exit(1)

# Humidity Sensor
DHTPin = 17
sensor = Adafruit_DHT.DHT11

class SimpleCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Camera App")

        # Set up video capture
        self.cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Error: Camera could not be initialized.")
            self.cap.release()
            return

        # Label for displaying the video feed
        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.humidity_label = Label(root, text="Humidity: -", font=("Arial", 14))
        self.humidity_label.pack()

        # Buttons
        self.add_button = Button(root, text="Add Item", command=self.start_capture_timer)
        self.add_button.pack()
        self.remove_button = Button(root, text="Remove Selected", command=self.remove_selected_item)
        self.remove_button.pack()

        # Start video feed update
        self.update_video_feed()
        self.update_humidity_display()

        table_frame = tk.Frame(root)
        table_frame.pack(fill=tk.BOTH, expand=True)

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

    def __del__(self):
        self.cap.release()  # Release the video capture when app is closed

    def update_humidity_display(self):
        humidity, temperature = Adafruit_DHT.read_retry(sensor, DHTPin)
        if humidity is not None:
            # humidity = round(humidity)
            # humidity_str = str(humidity)
            self.humidity_label.config(text=f"Humidity: {humidity:.2f}%")
        else:
            self.humidity_label.config(text="Failed to retrieve humidity data.")

        self.root.after(10000, self.update_humidity_display)  # Update every 2 seconds



class MotionHandler:
    def __init__(self):
        self.motion_count = 0
        self.running = True  # Control flag to stop the loops

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
            pi.set_servo_pulsewidth(servo_pin, 2500)  # Move to 180 degrees
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

    def open_camera_app(self):
        global root
        root = tk.Tk()
        app = SimpleCameraApp(root)

        # Start servo and humidity monitor in separate threads
        Thread(target=self.start_servo).start()
        Thread(target=self.start_humidity_monitor).start()
        
        root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        root.mainloop()

    def on_app_close(self): 
        global root, pi 
        self.running = False # Set the running flag to False 
        # Safely destroy the root window in the main thread 
        root.after(0, root.destroy) # Ensure this is in the main thread 
        pi.stop() # Stop the pigpio connection
        quit()

if __name__ == "__main__":
    try:
        motion_handler = MotionHandler()
        motion_thread = Thread(target=motion_handler.start_motion_sensor)
        motion_thread.start()

    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        # Perform cleanup actions if necessary
        motion_handler.on_app_close()