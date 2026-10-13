"""
Professional AI Virtual Keyboard & Air-Writing Pad (Fixed Imports)
Author: AI Collaborator
Description: Renders a fully visible QWERTY keyboard HUD with strict thumb-index 
             pinch detection for typing.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- CONFIGURATION CONSTANTS ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0
CLICK_COOLDOWN = 0.4      # Cooldown to prevent multi-triggering keys on a single pinch
PINCH_THRESHOLD = 28      # Strict distance threshold for selecting a key via pinch

# --- ONE-EURO FILTER FOR SMOOTH FINGERTIP TRACKING ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t, cutoff):
        r = 2 * np.pi * cutoff * (t - self.t_prev)
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        
        a_d = self.smoothing_factor(t, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def download_model_if_missing(path, url):
    if not os.path.exists(path):
        print("[INFO] Downloading MediaPipe Hand Landmarker model...")
        try:
            urllib.request.urlretrieve(url, path)
            print("[INFO] Model downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            exit(1)

# Clean and perfectly spaced QWERTY rows
KEYBOARD_ROWS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", "CLEAR", "SPACE"]
]

class Button:
    def __init__(self, pos, text, size):
        self.pos = pos
        self.size = size
        self.text = text

def draw_keyboard(frame, buttons, active_key=None, is_pinching=False):
    """Renders the virtual keyboard HUD ensuring all keys are fully visible."""
    for button in buttons:
        x, y = button.pos
        w, h = button.size
        
        if button.text == active_key:
            color = (0, 220, 100) if is_pinching else (0, 140, 255)
            border_color = (255, 255, 255)
        else:
            color = (50, 50, 50)
            border_color = (150, 150, 150)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
        
        font_scale = 0.4 if len(button.text) > 1 else 0.7
        text_x = x + 10 if len(button.text) > 1 else x + 15
        text_y = y + 37
        cv2.putText(frame, button.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

def main():
    download_model_if_missing(MODEL_PATH, MODEL_URL)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    buttons = []
    start_y = 310
    key_w, key_h, gap = 52, 52, 8

    for r_idx, row in enumerate(KEYBOARD_ROWS):
        row_length = len(row)
        if r_idx == 2:
            x_cursor = 25
            for key in row:
                w = 95 if key in ["CLEAR", "SPACE"] else key_w
                buttons.append(Button((x_cursor, start_y + r_idx * (key_h + gap)), key, (w, key_h)))
                x_cursor += w + gap
        else:
            total_row_width = (row_length * key_w) + ((row_length - 1) * gap)
            x_cursor = (640 - total_row_width) // 2
            for key in row:
                buttons.append(Button((x_cursor, start_y + r_idx * (key_h + gap)), key, (key_w, key_h)))
                x_cursor += key_w + gap

    init_time = time.time()
    filter_x = OneEuroFilter(init_time, 320, min_cutoff=0.8, beta=0.02)
    filter_y = OneEuroFilter(init_time, 240, min_cutoff=0.8, beta=0.02)

    typed_text = ""
    last_action_time = 0
    active_key = None

    print("[INFO] AI Virtual Keyboard running. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = time.time()
        frame_timestamp_ms = int((current_time - init_time) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        current_action = "HOVER"
        is_pinching = False
        
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]

            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)

            smooth_x = int(filter_x(current_time, index_x))
            smooth_y = int(filter_y(current_time, index_y))

            pinch_distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            is_pinching = pinch_distance < PINCH_THRESHOLD

            pointer_color = (0, 255, 0) if is_pinching else (0, 255, 255)
            cv2.circle(frame, (smooth_x, smooth_y), 8, pointer_color, cv2.FILLED)

            active_key = None
            for button in buttons:
                bx, by = button.pos
                bw, bh = button.size
                if bx < smooth_x < bx + bw and by < smooth_y < by + bh:
                    active_key = button.text
                    
                    if is_pinching:
                        current_action = "TYPING"
                        if current_time - last_action_time > CLICK_COOLDOWN:
                            if active_key == "CLEAR":
                                typed_text = ""
                            elif active_key == "SPACE":
                                typed_text += " "
                            else:
                                typed_text += active_key
                            last_action_time = current_time

        draw_keyboard(frame, buttons, active_key=active_key, is_pinching=is_pinching)

        cv2.rectangle(frame, (25, 220), (w - 25, 290), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(frame, (25, 220), (w - 25, 290), (200, 200, 200), 2)
        cv2.putText(frame, "Output Text:", (35, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, typed_text[-35:], (35, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Status: {current_action}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AI Virtual Keyboard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()