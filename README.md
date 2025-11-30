# Face Recognition Camera App

A real-time face detection, tracking, and recognition system built with Python, OpenCV, DeepFace, and Tkinter.

This application allows you to:

- Detect multiple faces using your webcam
- Save face embeddings linked to a custom name
- Compare live faces to saved identities
- Select faces by clicking on them
- Clear saved names or wipe all stored embeddings
- Use lightweight IoU-based face tracking

---

## 🚀 Features

- Live webcam video feed with face detection
- Face recognition using DeepFace embeddings
- Cosine similarity comparison
- Click-to-select facial bounding boxes
- Compare Mode toggle (ON/OFF)
- Persistent storage using `faces.json`
- Clear saved names with one button
- Optional sound effects via Pygame

---

## 📦 Dependencies

Install required libraries:

```bash
pip install opencv-python
pip install pillow\pip install deepface
pip install scipy
pip install pygame
pip install numpy
```

For convenience, if you create a `requirements.txt`, install everything with:

```bash
pip install -r requirements.txt
```

---

## 🛠 Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Add sounds

Place `good.mp3` and `perfect.mp3` inside a `/sounds` folder. The app will still work without audio.

---

## ▶️ Running the App

Run the program using:

```bash
python app.py
```

Replace `app.py` with your main file name if different.

---

## 📝 How to Use

### ✔ Saving a Face

1. Type a name into the text box
2. Click on a detected face in the video feed
3. Press **"Save Face Info"**

This saves a DeepFace embedding into `faces.json`.

### ✔ Compare Mode

Click **"Compare Faces: ON"** to begin live face matching.

Click again to turn it **OFF** and clear displayed labels.

### ✔ Clearing Saved Names

Press **"Clear Saved Names"** to wipe all embeddings in `faces.json`.

---

## 📁 File Structure

```
/your-project
 ├── app.py
 ├── faces.json
 ├── sounds/
 │    ├── good.mp3
 │    └── perfect.mp3
 ├── README.md
```

---

## 📄 Source Code

The full application code is located here:

➡️ `app.py`

---

## ⚠️ Notes

- A webcam is required
- DeepFace may take several seconds to initialize
- Default similarity threshold: **0.57**
- Haar Cascades are used for face detection (simple & fast)

---

