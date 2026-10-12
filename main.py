import time

import cv2
import mediapipe as mp

from gesture_detector import GestureStabilizer, detect_gesture, hand_metrics
from gesture_actions import (
    COOLDOWN_GESTURES,
    ONE_SHOT_GESTURES,
    cooldown_ready,
    control_mouse,
    perform_action,
    reset_mouse,
)
from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    ACTION_COOLDOWN,
    WINDOW_NAME,
    QUIT_KEY,
    DEBUG_KEY,
)
from utils import calculate_fps, draw_information


INDEX_FINGER_TIP = 8


def main():
    # ==========================================
    # MediaPipe Hands Setup
    # ==========================================

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    # ==========================================
    # Camera Setup
    # ==========================================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: could not open camera {CAMERA_INDEX}. "
              "Check that a webcam is connected and not used by another app.")
        hands.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"Running. Press '{QUIT_KEY}' in the video window to quit, "
          f"'{DEBUG_KEY}' to show/hide debug info.")

    previous_time = time.time()
    last_action_time = 0.0
    action = "Waiting..."
    stabilizer = GestureStabilizer()
    show_debug = False

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Error: failed to read a frame from the camera.")
                break

            # Mirror the image so movement feels natural, then convert
            # BGR (OpenCV) -> RGB (MediaPipe)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            current_time = time.time()
            raw_gesture = "NO HAND"
            hand_landmarks = None
            landmarks = None
            debug_text = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                raw_gesture = detect_gesture(landmarks)

                fingers, thumb_ratio = hand_metrics(landmarks)
                debug_text = (f"raw:{raw_gesture} fingers:{fingers} "
                              f"thumb:{thumb_ratio:+.2f}")
            else:
                reset_mouse()

            # Only act on a gesture that has been steady for a few frames
            gesture, changed = stabilizer.update(raw_gesture)

            if gesture == "ONE FINGER":
                if landmarks is not None:
                    # Cursor follows the fingertip every frame (no cooldown)
                    tip_x, tip_y = landmarks[INDEX_FINGER_TIP]
                    control_mouse(tip_x, tip_y)
                action = perform_action(gesture, frame, current_time)

            elif gesture in COOLDOWN_GESTURES:
                if gesture in ONE_SHOT_GESTURES and not changed:
                    # Already fired for this hold; lower the hand to repeat
                    pass
                elif cooldown_ready(
                    last_action_time, current_time, ACTION_COOLDOWN
                ):
                    action = perform_action(gesture, frame, current_time)
                    last_action_time = current_time
                else:
                    action = "Cooldown"

            else:
                action = perform_action(gesture, frame, current_time)

            # Draw after the action so screenshots stay clean
            if hand_landmarks is not None:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            previous_time, fps = calculate_fps(previous_time)
            frame = draw_information(
                frame,
                gesture,
                action,
                fps,
                debug_text if show_debug else None
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(QUIT_KEY):
                break

            if key == ord(DEBUG_KEY):
                show_debug = not show_debug

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
