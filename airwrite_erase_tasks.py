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

# Brushes
DRAW_COLOR = (0, 0, 255)
DRAW_THICKNESS = 6

ERASE_THICKNESS = 30  # bigger = easier erasing

SMOOTHING_ALPHA = 0.2


def finger_up(hand, tip_id, pip_id):
    return hand[tip_id].y < hand[pip_id].y


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    # Speed boost (optional but recommended)
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

        if canvas is None:
            canvas = np.zeros_like(frame)

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

            # Mode selection
            if index_up and middle_up:
                mode = "ERASE"
            elif index_up:
                mode = "DRAW"
            else:
                mode = "PAUSE"

            mode_text = mode

            # Use index fingertip as pointer
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
                    cv2.line(canvas, prev_pt, (x, y), DRAW_COLOR, DRAW_THICKNESS)
                prev_pt = (x, y)

            elif mode == "ERASE":
                cv2.circle(frame, (x, y), 12, (0, 0, 0), -1)
                if prev_pt is not None:
                    # Erase by drawing black on the canvas
                    cv2.line(canvas, prev_pt, (x, y), (0, 0, 0), ERASE_THICKNESS)
                prev_pt = (x, y)

            else:  # PAUSE
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                prev_pt = None

        out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

        cv2.putText(out, f"Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, "q=quit  c=clear  s=save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AirWrite (Draw + Erase)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("s"):
            cv2.imwrite("airwrite_canvas.png", canvas)
            print("Saved: airwrite_canvas.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()