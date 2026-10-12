import time

import cv2


def calculate_fps(previous_time):
    """Return (current_time, fps) given the timestamp of the previous frame."""
    current_time = time.time()
    elapsed = current_time - previous_time

    # Guard against division by zero on the very first frame
    fps = 1 / elapsed if elapsed > 0 else 0

    return current_time, fps


def draw_information(frame, gesture, action, fps, debug_text=None):
    """
    Draw a small info panel (gesture, action, FPS) on the frame.
    If `debug_text` is given, it is drawn as an extra line.
    """

    panel_bottom = 145 if debug_text is None else 175

    cv2.rectangle(
        frame,
        (20, 20),
        (350, panel_bottom),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Action: {action}",
        (35, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (35, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if debug_text is not None:
        cv2.putText(
            frame,
            debug_text,
            (35, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return frame
