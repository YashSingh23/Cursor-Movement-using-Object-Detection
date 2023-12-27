import cv2
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

contour_counts = []

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Motion Detection App")

        self.root.geometry("500x250")

        self.vs = None
        self.panel = None
        self.stop_event = None

        self.create_widgets()

    def create_widgets(self):
        self.btn_start_stop = tk.Button(self.root, text="Start Motion Detection", command=self.toggle_object_detection, bd = 5, font=('Times', 18))
        self.btn_start_stop.pack(pady=10)

        self.btn_start_stop.place(x=150, y=20)

        self.my_label = tk.Label(self.root, text="Contour Count = 0", bd=5, font=('Times', 24))
        self.my_label.pack(pady=10)

        self.my_label.place(x=150, y=80)
        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

    def toggle_object_detection(self):
        if self.btn_start_stop["text"] == "Start Motion Detection":
            self.vs = cv2.VideoCapture(0)
            self.stop_event = False
            self.btn_start_stop["text"] = "Stop Motion Detection"
            Thread(target=self.detect_objects, args=()).start()
        else:
            self.stop_event = True
            self.btn_start_stop["text"] = "Start Motion Detection"



    def detect_objects(self):
        cap = cv2.VideoCapture(0)
        fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            fgmask = fgbg.apply(frame)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = 0

            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                contour_count += 1
                my_text = f"Contour Count = {contour_count}"
                self.my_label.config(text=my_text)

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Foreground Mask', fgmask)

            key = cv2.waitKey(30)
            if key == 27 or self.btn_start_stop["text"] == "Start Motion Detection":
                break

            contour_counts.append(contour_count)

        cap.release()
        cv2.destroyAllWindows()
        self.data_visualization()


    def update(self):
        ret, frame = self.vs.read()
        self.root.after(10, self.update)

    def data_visualization(self):
        plt.plot(contour_counts)
        plt.title('Contour Counts over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Contour Count')
        plt.show()


root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()