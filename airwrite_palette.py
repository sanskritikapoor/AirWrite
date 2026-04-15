import time
import os
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Landmarks
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

SMOOTHING_ALPHA = 0.2

# Brushes
draw_color = (0, 0, 255)
draw_thickness = 6
erase_thickness = 30
MIN_THICKNESS = 1
MAX_THICKNESS = 50

# UI (right side)
UI_BOX = 28
UI_PAD = 8
UI_TOP = 10
UI_RIGHT_MARGIN = 10
SHOW_STATUS_TEXT = True

# WORLD canvas
WORLD_W = 3000
WORLD_H = 3000
WORLD_BG = 255

# View transform: screen = zoom*world + offset
zoom = 1.0
MIN_ZOOM = 0.4
MAX_ZOOM = 3.0

# 2-hand zoom tuning
TWO_HAND_DEADZONE_PX = 6      # ignore tiny changes in fingertip distance
TWO_HAND_ZOOM_SENS = 0.004    # zoom change per pixel distance delta (tune this)

# Color selection (middle finger)
COLOR_SELECT_COOLDOWN = 0.35


def finger_up(hand, tip_id, pip_id):
    return hand[tip_id].y < hand[pip_id].y


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def save_canvas_png(world_canvas, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("airwrite_%Y-%m-%d_%H-%M-%S.png")
    path = os.path.join(folder, filename)
    ok = cv2.imwrite(path, world_canvas)
    print(("Saved: " if ok else "Save failed: "), path)


def palm_center_px(hand, w, h):
    palm_points = [WRIST, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    px = int(np.mean([hand[i].x for i in palm_points]) * w)
    py = int(np.mean([hand[i].y for i in palm_points]) * h)
    return px, py


def screen_to_world(x, y, offx, offy, z):
    wx = int((x - offx) / z)
    wy = int((y - offy) / z)
    return wx, wy


def world_view(world_canvas, out_w, out_h, offx, offy, z):
    M = np.float32([[z, 0, offx], [0, z, offy]])
    view = cv2.warpAffine(
        world_canvas, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(WORLD_BG, WORLD_BG, WORLD_BG),
    )
    return view


def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def main():
    global draw_color, draw_thickness, zoom

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,   # <-- important for 2-hand zoom
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "AirWrite (Palette)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    world = np.ones((WORLD_H, WORLD_W, 3), dtype=np.uint8) * WORLD_BG

    offset_x, offset_y = 0, 0

    prev_world_pt = None
    prev_palm_move = None

    # 2-hand zoom state
    prev_twohand_dist = None

    # smoothed index fingertip for hand0 drawing
    smooth_x, smooth_y = None, None

    last_color_select_time = 0.0

    palette = [
        ("R", (0, 0, 255)), ("G", (0, 255, 0)), ("B", (255, 0, 0)),
        ("K", (0, 0, 0)), ("W", (255, 255, 255)), ("Y", (0, 255, 255)),
        ("C", (255, 255, 0)), ("M", (255, 0, 255)), ("O", (0, 165, 255)),
        ("P", (128, 0, 128)), ("N", (19, 69, 139)), ("S", (128, 128, 128)),
    ]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ---- palette grid (right) ----
        cols = 2
        rows = int(np.ceil(len(palette) / cols))
        panel_w = cols * UI_BOX + (cols - 1) * UI_PAD
        panel_x2 = w - UI_RIGHT_MARGIN
        panel_x1 = panel_x2 - panel_w
        panel_y1 = UI_TOP

        color_rects = []
        for i, (name, color) in enumerate(palette):
            r = i // cols
            c = i % cols
            x1 = panel_x1 + c * (UI_BOX + UI_PAD)
            y1 = panel_y1 + r * (UI_BOX + UI_PAD)
            x2 = x1 + UI_BOX
            y2 = y1 + UI_BOX
            color_rects.append((name, color, (x1, y1, x2, y2)))

        controls_y = panel_y1 + rows * (UI_BOX + UI_PAD) + UI_PAD
        minus_rect = (panel_x1, controls_y, panel_x1 + UI_BOX, controls_y + UI_BOX)
        plus_rect = (panel_x1 + UI_BOX + UI_PAD, controls_y, panel_x1 + 2 * UI_BOX + UI_PAD, controls_y + UI_BOX)

        # ---- MediaPipe ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mode_text = "NO HAND"

        hands = result.hand_landmarks if result.hand_landmarks else []

        # ---------- TWO HAND ZOOM ----------
        if len(hands) >= 2:
            h0, h1 = hands[0], hands[1]

            x0, y0 = lm_to_px(h0[INDEX_TIP], w, h)
            x1, y1 = lm_to_px(h1[INDEX_TIP], w, h)

            cv2.circle(frame, (x0, y0), 8, (0, 0, 0), -1)
            cv2.circle(frame, (x1, y1), 8, (0, 0, 0), -1)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 0), 2)

            dist = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

            if prev_twohand_dist is not None:
                dd = dist - prev_twohand_dist  # + if hands moved apart
                if abs(dd) > TWO_HAND_DEADZONE_PX:
                    # Hands apart -> zoom IN
                    zoom = clamp(zoom + dd * TWO_HAND_ZOOM_SENS, MIN_ZOOM, MAX_ZOOM)

            prev_twohand_dist = dist
            mode_text = "ZOOM (2H)"

            # disable drawing/move while 2-hand zooming
            prev_world_pt = None
            prev_palm_move = None
            smooth_x, smooth_y = None, None

        else:
            # reset two-hand zoom memory when not using it
            prev_twohand_dist = None

            # ---------- ONE HAND MODES ----------
            if len(hands) == 0:
                prev_world_pt = None
                prev_palm_move = None
                smooth_x, smooth_y = None, None
            else:
                hand = hands[0]

                index_up = finger_up(hand, INDEX_TIP, INDEX_PIP)
                middle_up = finger_up(hand, MIDDLE_TIP, MIDDLE_PIP)
                ring_up = finger_up(hand, RING_TIP, RING_PIP)
                pinky_up = finger_up(hand, PINKY_TIP, PINKY_PIP)
                thumb_up = finger_up(hand, THUMB_TIP, THUMB_IP)
                all_up = index_up and middle_up and ring_up and pinky_up and thumb_up

                # index fingertip smoothing (screen)
                tip = hand[INDEX_TIP]
                raw_x, raw_y = int(tip.x * w), int(tip.y * h)
                if smooth_x is None:
                    smooth_x, smooth_y = raw_x, raw_y
                else:
                    smooth_x = int((1 - SMOOTHING_ALPHA) * smooth_x + SMOOTHING_ALPHA * raw_x)
                    smooth_y = int((1 - SMOOTHING_ALPHA) * smooth_y + SMOOTHING_ALPHA * raw_y)
                fx, fy = smooth_x, smooth_y

                # middle fingertip for color selection
                m_tip = hand[MIDDLE_TIP]
                mx, my = int(m_tip.x * w), int(m_tip.y * h)

                # mode selection
                if all_up:
                    mode = "MOVE"
                elif index_up and middle_up:
                    mode = "ERASE"
                elif index_up:
                    mode = "DRAW"
                else:
                    mode = "PAUSE"
                mode_text = mode

                # middle finger selection
                now = time.time()
                if (now - last_color_select_time) > COLOR_SELECT_COOLDOWN:
                    for _, color, rect in color_rects:
                        if in_rect(mx, my, rect):
                            draw_color = color
                            last_color_select_time = now
                            break
                    if in_rect(mx, my, minus_rect):
                        draw_thickness = clamp(draw_thickness - 1, MIN_THICKNESS, MAX_THICKNESS)
                        last_color_select_time = now
                    elif in_rect(mx, my, plus_rect):
                        draw_thickness = clamp(draw_thickness + 1, MIN_THICKNESS, MAX_THICKNESS)
                        last_color_select_time = now

                pcx, pcy = palm_center_px(hand, w, h)

                if mode == "MOVE":
                    if prev_palm_move is not None:
                        offset_x += (pcx - prev_palm_move[0])
                        offset_y += (pcy - prev_palm_move[1])
                    prev_palm_move = (pcx, pcy)
                    prev_world_pt = None

                elif mode == "DRAW":
                    wx, wy = screen_to_world(fx, fy, offset_x, offset_y, zoom)
                    if prev_world_pt is not None:
                        cv2.line(world, prev_world_pt, (wx, wy), draw_color, draw_thickness)
                    prev_world_pt = (wx, wy)
                    prev_palm_move = None

                elif mode == "ERASE":
                    wx, wy = screen_to_world(fx, fy, offset_x, offset_y, zoom)
                    if prev_world_pt is not None:
                        cv2.line(world, prev_world_pt, (wx, wy), (255, 255, 255), erase_thickness)
                    prev_world_pt = (wx, wy)
                    prev_palm_move = None

                else:
                    prev_world_pt = None
                    prev_palm_move = None

                # cursors
                cv2.circle(frame, (fx, fy), 6, (0, 255, 0) if mode == "DRAW" else (255, 0, 0), -1)
                cv2.circle(frame, (mx, my), 6, (255, 255, 255), -1)
                cv2.circle(frame, (mx, my), 6, (0, 0, 0), 2)

        # ---- UI draw ----
        for _, color, rect in color_rects:
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            if color == draw_color:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 2)

        for rect, label in [(minus_rect, "-"), (plus_rect, "+")]:
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, label, (x1 + 8, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # ---- render world ----
        view = world_view(world, w, h, offset_x, offset_y, zoom)
        out = cv2.addWeighted(frame, 0.70, view, 0.30, 0)

        if SHOW_STATUS_TEXT:
            label = f"Gesture: {mode_text}   Zoom: {zoom:.2f}   T: {draw_thickness}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            tx = (w - tw) // 2
            ty = h - 15
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        cv2.imshow(window_name, out)

        key = cv2.waitKeyEx(1)
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("c"), ord("C")):
            world[:] = WORLD_BG
            offset_x, offset_y = 0, 0
            zoom = 1.0
        elif key in (ord("s"), ord("S")):
            save_canvas_png(world)
        elif key == ord("0"):
            zoom = 1.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()