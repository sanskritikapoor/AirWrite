import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download this model (next step) and put it in the same folder as this script:
MODEL_PATH = "models/hand_landmarker.task"

def main():
    # Create the HandLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe Image expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # hand is a list of 21 landmarks with x,y in [0,1]
                for lm in hand:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # index fingertip id = 8
                tip = hand[8]
                x8, y8 = int(tip.x * w), int(tip.y * h)
                cv2.circle(frame, (x8, y8), 10, (0, 0, 255), -1)

        cv2.imshow("Hands Test (Tasks API)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()