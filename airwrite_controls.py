import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Landmark indices
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10

SMOOTHING_ALPHA = 0.2

# Defaults
draw_color = (0, 0, 255)     # red
draw_thickness = 6
erase_thickness = 30

MIN_THICKNESS = 1
MAX_THICKNESS = 50


def finger_up(hand, tip_id, pip_id):
    return hand[tip_id].y < hand[pip_id].y


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    global draw_color, draw_thickness, erase_thickness

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    canvas = None
    prev_pt = None
    smooth_x, smooth_y = None, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # WHITE CANVAS so black ink is visible
        if canvas is None:
            canvas = np.ones_like(frame, dtype=np.uint8) * 255

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mode_text = "NO HAND"

        if not result.hand_landmarks:
            prev_pt = None
            smooth_x, smooth_y = None, None
        else:
            hand = result.hand_landmarks[0]

            index_up = finger_up(hand, INDEX_TIP, INDEX_PIP)
            middle_up = finger_up(hand, MIDDLE_TIP, MIDDLE_PIP)

            if index_up and middle_up:
                mode = "ERASE"
            elif index_up:
                mode = "DRAW"
            else:
                mode = "PAUSE"

            mode_text = mode

            tip = hand[INDEX_TIP]
            raw_x, raw_y = int(tip.x * w), int(tip.y * h)

            # smoothing
            if smooth_x is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int((1 - SMOOTHING_ALPHA) * smooth_x + SMOOTHING_ALPHA * raw_x)
                smooth_y = int((1 - SMOOTHING_ALPHA) * smooth_y + SMOOTHING_ALPHA * raw_y)

            x, y = smooth_x, smooth_y

            if mode == "DRAW":
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, (x, y), draw_color, draw_thickness)
                prev_pt = (x, y)

            elif mode == "ERASE":
                # Erase by painting WHITE on the white canvas
                cv2.circle(frame, (x, y), 12, (0, 0, 0), -1)
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, (x, y), (255, 255, 255), erase_thickness)
                prev_pt = (x, y)

            else:
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                prev_pt = None

        # Blend lightly so the camera view doesn't get washed out
        out = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        # UI text
        cv2.putText(out, f"Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.putText(out, f"Brush thickness={draw_thickness}  Eraser={erase_thickness}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(out, "Click this window, then keys: r/g/b/k/w, [, ], -/+, c, s, q",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.imshow("AirWrite (Controls)", out)

        # Use waitKeyEx (more reliable on Windows)
        key = cv2.waitKeyEx(1)

        # quit
        if key in (ord("q"), ord("Q")):
            break

        # save / clear
        elif key in (ord("c"), ord("C")):
            canvas[:] = 255  # clear to white
        elif key in (ord("s"), ord("S")):
            cv2.imwrite("airwrite_canvas.png", canvas)
            print("Saved: airwrite_canvas.png")

        # colors
        elif key in (ord("r"), ord("R")):
            draw_color = (0, 0, 255)
        elif key in (ord("g"), ord("G")):
            draw_color = (0, 255, 0)
        elif key in (ord("b"), ord("B")):
            draw_color = (255, 0, 0)
        elif key in (ord("k"), ord("K")):
            draw_color = (0, 0, 0)
        elif key in (ord("w"), ord("W")):
            draw_color = (255, 255, 255)

        # thickness (brackets)
        elif key == ord("["):
            draw_thickness = clamp(draw_thickness - 1, MIN_THICKNESS, MAX_THICKNESS)
        elif key == ord("]"):
            draw_thickness = clamp(draw_thickness + 1, MIN_THICKNESS, MAX_THICKNESS)

        # eraser thickness
        elif key == ord("-"):
            erase_thickness = clamp(erase_thickness - 2, 5, 120)
        elif key in (ord("+"), ord("=")):  # '=' is shift+'+'
            erase_thickness = clamp(erase_thickness + 2, 5, 120)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()