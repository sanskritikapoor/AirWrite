# AirWrite
# AirWrite: Virtual Drawing Using Hand Gestures

AirWrite is a contactless virtual drawing application that allows users to draw, erase, pan, and zoom on a digital canvas using only hand gestures detected through a webcam. Powered by MediaPipe for robust hand landmark detection and OpenCV for real-time video handling, AirWrite offers an intuitive, interactive experience without the need for physical input devices.

---

## Features

- ✍️ **Draw**: Use your index finger to draw smooth lines on a world canvas.
- 🧹 **Erase**: Erase unwanted strokes with a simple gesture.
- 🖐 **Pan & Zoom**: Move and scale your canvas using natural hand gestures.
- 🎨 **Customizable Tools**: Select colors and brush sizes via an on-screen palette.
- 💾 **Save Artwork**: Export your creations with a keyboard shortcut.
- ⚡ **Real-time Performance**: Instant feedback on interactions.

---

## System Requirements

- **Python 3.7+**
- **OpenCV-Python**
- **MediaPipe**
- Webcam (internal or external)

---

## Installation

1. **Clone this repository:**

    ```bash
    git clone https://github.com/yourusername/AirWrite.git
    cd AirWrite
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Connect your webcam.**
2. **Run the main script:**

    ```bash
    python airwrite_basic_tasks.py
    ```

3. **Controls:**

    - **Draw:** Raise only your index finger.
    - **Erase:** Raise index and middle finger.
    - **Pan:** Open palm (all fingers).
    - **Zoom:** Use both hands — move index fingertips apart/together.
    - **Change Color/Brush:** Use the palette overlay, hover and raise middle finger to select.
    - **Save Art/Clear Canvas:** Use provided keyboard shortcuts (see script/documentation).

---

## Project Structure

```
AirWrite/
│
├── airwrite_basic_tasks.py     # Main application script
├── airwrite_palette.py         # Palette UI code
├── airwrite_erase_tasks.py     # Eraser mode logic
├── airwrite_controls.py        # Gesture handling
├── models/                     # Any ML models
├── outputs/                    # Saved drawings
├── requirements.txt            # Python dependencies
└── ...
```

---

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'cv2'` or `mediapipe`, ensure you've installed all requirements.
- Make sure your webcam is properly connected and not used by another application.
- For best results, use against a plain background under good lighting.


## Acknowledgments

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [OpenCV](https://opencv.org/)

---

Created by Sanskriti Kapoor
