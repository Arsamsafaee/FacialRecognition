import tkinter as tk
from tkinter import *
import json
import os
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from scipy.spatial.distance import cosine
from pygame import mixer
import time
import threading
import numpy as np

# ---------- Helper: IoU ----------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0
    return interArea / unionArea

# ---------- Centralized JSON I/O ----------
def save_faces(data):
    serial = {}
    for k, v in data.items():
        try:
            serial[k] = list(map(float, v))
        except:
            serial[k] = v
    with open("faces.json", "w") as f:
        json.dump(serial, f, indent=2)

def load_faces():
    if not os.path.exists("faces.json"):
        return {}
    try:
        with open("faces.json", "r") as f:
            data = json.load(f)
    except:
        return {}

    out = {}
    for name, emb in data.items():
        try:
            out[name] = np.array(emb, dtype=float)
        except:
            continue
    return out

class CameraApp:
    def __init__(self, root, title='Camera App'):
        self.root = root
        self.root.title(title)

        self.compare_mode = False  # ON/OFF switch state

        # Entry field (for naming saved faces)
        self.entry_widget = Entry(root, width=30)
        self.entry_widget.pack(pady=8)

        # Buttons
        button_frame = Frame(root)
        button_frame.pack(pady=5)

        self.start_button = Button(button_frame, text='Set Name (for Save)', command=self.save_name)
        self.start_button.pack(side=LEFT, padx=5)

        self.save_face_button = Button(button_frame, text='Save Face Info', command=self.save_face_info)
        self.save_face_button.pack(side=LEFT, padx=5)

        self.compare_button = Button(button_frame, text='Compare Faces: OFF', command=self.toggle_compare_mode)
        self.compare_button.pack(side=LEFT, padx=5)

        self.new_person_button = Button(button_frame, text='Clear Name', command=self.clear_name)
        self.new_person_button.pack(side=LEFT, padx=5)

        # NEW BUTTON → CLEAR FACES.JSON
        self.clear_file_button = Button(button_frame, text='Clear Saved Names', command=self.clear_saved_faces)
        self.clear_file_button.pack(side=LEFT, padx=5)

        # Canvas for video
        self.canvas_w = 640
        self.canvas_h = 480
        self.canvas = Canvas(self.root, width=self.canvas_w, height=self.canvas_h)
        self.canvas.pack()

        # click selection for saving
        self.canvas.bind("<Button-1>", self.select_face)

        # audio (optional)
        try:
            mixer.init()
            self.good = mixer.Sound('sounds/good.mp3')
            self.perfect = mixer.Sound('sounds/perfect.mp3')
        except:
            self.good = None
            self.perfect = None

        # state
        self.entered_text = ""
        self.current_frame = None
        self.detected_boxes = []
        self.selected_face_index = None
        self.tracks = {}   # tid -> {"bbox":(), "last_seen": t, "name": str}
        self.next_id = 0
        self.MAX_TRACK_AGE = 0.6

        # haar face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera")

        # start updating
        self.update_frame()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =========================================================
    #   NEW FUNCTION: CLEAR faces.json
    # =========================================================
    def clear_saved_faces(self):
        save_faces({})
        print("All saved face data cleared.")

    # =========================================================
    #   COMPARE MODE SWITCH
    # =========================================================
    def toggle_compare_mode(self):
        self.compare_mode = not self.compare_mode

        if self.compare_mode:
            self.compare_button.config(text="Compare Faces: ON")
            print("Compare mode ENABLED.")
        else:
            self.compare_button.config(text="Compare Faces: OFF")
            print("Compare mode DISABLED.")

            # Clear all names when turning off
            for tid in self.tracks:
                self.tracks[tid]["name"] = ""

    # =========================================================
    #   UI HELPERS
    # =========================================================
    def save_name(self):
        name = self.entry_widget.get().strip()
        if name:
            self.entered_text = name
            print("Name set to:", name)
        else:
            print("Please enter a name first.")

    def clear_name(self):
        self.entry_widget.delete(0, END)
        self.entered_text = ""
        print("Name cleared.")

    def select_face(self, event):
        for i, (x, y, w, h) in enumerate(self.detected_boxes):
            if x <= event.x <= x+w and y <= event.y <= y+h:
                self.selected_face_index = i
                print(f"Selected face {i+1}")
                return
        self.selected_face_index = None
        print("No face selected.")

    # =========================================================
    #   SAVE NEW FACE EMBEDDING
    # =========================================================
    def save_face_info(self):
        if self.current_frame is None:
            print("No frame available.")
            return
        if self.selected_face_index is None:
            print("Click a face first.")
            return
        if not self.entered_text:
            print("Set a name first.")
            return

        x, y, w, h = self.detected_boxes[self.selected_face_index]
        H, W, _ = self.current_frame.shape

        face_crop = self.current_frame[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]

        try:
            emb = DeepFace.represent(face_crop, enforce_detection=False)[0]['embedding']
            emb = np.array(emb, dtype=float)
        except Exception as e:
            print("Embedding error:", e)
            return

        faces = load_faces()
        faces[self.entered_text] = emb
        save_faces(faces)

        print("Saved embedding for:", self.entered_text)

    # =========================================================
    #   FACE COMPARISON LOGIC
    # =========================================================
    def compare_all_faces(self):
        if not self.compare_mode:
            return  # Only compare when ON

        if self.current_frame is None:
            return

        saved_faces = load_faces()
        if not saved_faces:
            return

        threshold = 0.57
        track_items = list(self.tracks.items())

        for tid, data in track_items:
            x, y, w, h = data["bbox"]
            H, W, _ = self.current_frame.shape
            face_crop = self.current_frame[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]

            try:
                emb = DeepFace.represent(face_crop, enforce_detection=False)[0]['embedding']
                emb = np.array(emb, dtype=float)
            except:
                continue

            best_sim = -1
            best_name = ""

            for name, saved_emb in saved_faces.items():
                try:
                    sim = 1 - cosine(emb, saved_emb)
                except:
                    sim = -1

                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim > threshold:
                self.tracks[tid]["name"] = best_name

    # =========================================================
    #   FRAME UPDATE LOOP
    # =========================================================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = list(self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60)))

        # -------- update tracking --------
        new_tracks = {}
        used_ids = set()

        for det in detected:
            x, y, w, h = map(int, det)

            best_tid = None
            best_iou_val = 0

            for tid, t in self.tracks.items():
                i = iou((x,y,w,h), t["bbox"])
                if i > best_iou_val:
                    best_iou_val = i
                    best_tid = tid

            if best_tid is not None and best_iou_val >= 0.25 and best_tid not in used_ids:
                new_tracks[best_tid] = {
                    "bbox": (x,y,w,h),
                    "last_seen": time.time(),
                    "name": self.tracks[best_tid]["name"]
                }
                used_ids.add(best_tid)
            else:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = {
                    "bbox": (x,y,w,h),
                    "last_seen": time.time(),
                    "name": ""  # no name initially
                }
                used_ids.add(tid)

        for tid, t in self.tracks.items():
            if tid not in used_ids:
                if time.time() - t["last_seen"] < self.MAX_TRACK_AGE:
                    new_tracks[tid] = t

        self.tracks = new_tracks
        sorted_tids = sorted(self.tracks.keys())
        self.detected_boxes = [self.tracks[tid]["bbox"] for tid in sorted_tids]

        # -------- run comparison only when ON --------
        if self.compare_mode:
            self.compare_all_faces()

        # -------- draw output --------
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
        self.canvas.image = imgtk

        for tid in sorted_tids:
            x, y, w, h = self.tracks[tid]["bbox"]
            name = self.tracks[tid]["name"] if self.compare_mode else ""

            self.canvas.create_rectangle(x, y, x+w, y+h, outline="blue", width=3)
            self.canvas.create_text(x+2, y-10, text=name, fill="orange", anchor=SW,
                                    font=("Arial", 14, "bold"))

        self.root.after(30, self.update_frame)

    def on_close(self):
        try:
            self.cap.release()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
