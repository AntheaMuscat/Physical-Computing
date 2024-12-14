import tkinter as tk
from tkinter import Button, Label
import cv2
from PIL import Image, ImageTk
import datetime

class SimpleCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Camera App")

        # Set up video capture
        self.cap = cv2.VideoCapture(0)

        # Label for displaying the video feed
        self.img_label = Label(root)
        self.img_label.pack()

        # Buttons
        self.add_button = Button(root, text="Add Item", command=self.start_capture_timer)
        self.add_button.pack()

        self.remove_button = Button(root, text="Remove Item", command=self.remove_item)
        self.remove_button.pack()

        self.update_button = Button(root, text="Update Item", command=self.update_item)
        self.update_button.pack()

        # Start video feed update
        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk

        self.root.after(10, self.update_video_feed)  # Refresh video feed

    def start_capture_timer(self):
        # Disable the Add Item button temporarily
        self.add_button.config(state=tk.DISABLED)
        self.countdown(5)

    def countdown(self, remaining_time):
        if remaining_time > 0:
            print(f"Taking photo in {remaining_time} seconds...")
            self.root.after(1000, self.countdown, remaining_time - 1)
        else:
            self.capture_and_save_image()
            self.add_button.config(state=tk.NORMAL)  # Re-enable the Add Item button

    def capture_and_save_image(self):
        ret, frame = self.cap.read()
        if ret:
            filename = f"captured_image.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")

    def remove_item(self):
        print("Remove Item button pressed")

    def update_item(self):
        print("Update Item button pressed")

    def __del__(self):
        self.cap.release()  # Release the video capture when app is closed


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleCameraApp(root)
    root.mainloop()
