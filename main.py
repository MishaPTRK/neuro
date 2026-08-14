import time
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONF_THRES = 0.35
IMGSZ = 640
FRAME_W = 1280
FRAME_H = 720

COLOR_PRIMARY = (0, 255, 140)
COLOR_TEXT_DIM = (0, 180, 100)
COLOR_BG_PANEL = (10, 10, 10)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX


def draw_transparent_rect(img, pt1, pt2, color, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_corner_frame(img, margin=18, length=34, thickness=2, color=COLOR_PRIMARY):
    h, w = img.shape[:2]
    corners = [
        (margin, margin, 1, 1),
        (w - margin, margin, -1, 1),
        (margin, h - margin, 1, -1),
        (w - margin, h - margin, -1, -1),
    ]
    for x, y, dx, dy in corners:
        cv2.line(img, (x, y), (x + dx * length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + dy * length), color, thickness)


def draw_bracket_box(img, x1, y1, x2, y2, color, thickness=2, length_ratio=0.22):
    w = x2 - x1
    h = y2 - y1
    lx = max(6, int(w * length_ratio))
    ly = max(6, int(h * length_ratio))

    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)


def put_text_with_bg(img, text, org, font=FONT, scale=0.5, color=COLOR_PRIMARY,
                     thickness=1, pad=4, bg_alpha=0.4):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    draw_transparent_rect(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                          COLOR_BG_PANEL, bg_alpha)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def run():
    model = YOLO(MODEL_PATH)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    fps_buffer = deque(maxlen=30)
    frame_idx = 0

    window_name = "Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            t0 = time.time()
            frame = picam2.capture_array()

            h, w = frame.shape[:2]
            frame_idx += 1

            results = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)[0]

            boxes = []
            for b in results.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                label = model.names.get(cls_id, str(cls_id))
                boxes.append({"xyxy": (x1, y1, x2, y2), "label": label, "conf": conf})

            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                draw_bracket_box(frame, x1, y1, x2, y2, COLOR_PRIMARY, thickness=2)
                tag = f"{b['label'].upper()} {b['conf'] * 100:.0f}%"
                put_text_with_bg(frame, tag, (x1, max(20, y1 - 8)), scale=0.5, color=COLOR_PRIMARY)

            draw_corner_frame(frame, color=COLOR_PRIMARY)

            dt = time.time() - t0
            fps_buffer.append(1.0 / dt if dt > 0 else 0)
            fps = sum(fps_buffer) / len(fps_buffer)

            put_text_with_bg(frame, f"FPS: {fps:5.1f}", (18, 30), font=FONT_MONO,
                             scale=0.6, color=COLOR_PRIMARY)
            put_text_with_bg(frame, f"OBJ: {len(boxes)}", (18, 56), scale=0.5,
                             color=COLOR_TEXT_DIM)

            info_line = f"FRAME {frame_idx:06d}   RES {w}x{h}"
            draw_transparent_rect(frame, (0, h - 30), (w, h), COLOR_BG_PANEL, alpha=0.45)
            cv2.putText(frame, info_line, (18, h - 10), FONT, 0.5, COLOR_TEXT_DIM, 1,
                        cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
