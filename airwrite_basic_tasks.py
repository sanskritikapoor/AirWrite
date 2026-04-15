import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

INDEX_TIP = 8
INDEX_PIP = 6

BRUSH_COLOR = (0, 0, 255)
BRUSH_THICKNESS = 6

SMOOTHING_ALPHA = 0.2


def index_finger_up(hand_landmarks):
    tip = hand_landmarks[INDEX_TIP]
    pip = hand_landmarks[INDEX_PIP]
    return tip.y < pip.y


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # IMPORTANT (faster)
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

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

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in milliseconds (required for VIDEO)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mode_text = "NO HAND"
        draw_mode = False

        if not result.hand_landmarks:
            prev_pt = None
            smooth_x, smooth_y = None, None
        else:
            hand = result.hand_landmarks[0]

            draw_mode = index_finger_up(hand)
            mode_text = "DRAW" if draw_mode else "PAUSE"

            tip = hand[INDEX_TIP]
            raw_x, raw_y = int(tip.x * w), int(tip.y * h)

            # smoothing
            if smooth_x is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int((1 - SMOOTHING_ALPHA) * smooth_x + SMOOTHING_ALPHA * raw_x)
                smooth_y = int((1 - SMOOTHING_ALPHA) * smooth_y + SMOOTHING_ALPHA * raw_y)

            x, y = smooth_x, smooth_y

            pointer_color = (0, 255, 0) if draw_mode else (255, 0, 0)
            cv2.circle(frame, (x, y), 8, pointer_color, -1)

            if draw_mode:
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, (x, y), BRUSH_COLOR, BRUSH_THICKNESS)
                prev_pt = (x, y)
            else:
                prev_pt = None

        out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

        cv2.putText(out, f"Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, "q=quit  c=clear  s=save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AirWrite (Gesture - Fast)", out)

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