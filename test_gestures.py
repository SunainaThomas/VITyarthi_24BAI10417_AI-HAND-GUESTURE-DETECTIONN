"""
Tests for gesture detection and gesture actions.

The detector tests use synthetic MediaPipe-style landmarks: 21 (x, y) points
in normalised image coordinates (y grows downwards). No camera or MediaPipe
is needed.

Run with either:
    pytest -v
    python test_gestures.py
"""

import math
import os
import tempfile
from unittest import mock

import numpy as np

import gesture_actions
from gesture_actions import (
    COOLDOWN_GESTURES,
    ONE_SHOT_GESTURES,
    cooldown_ready,
    map_to_screen,
    perform_action,
    smooth_position,
)
from gesture_detector import (
    GestureStabilizer,
    detect_gesture,
    distance,
    hand_metrics,
)


# ==========================================
# Landmark fixture builder
# ==========================================

# Index of (MCP, PIP, DIP, TIP) landmarks for each non-thumb finger
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGER_X = {"index": 0.42, "middle": 0.50, "ring": 0.58, "pinky": 0.65}

# Wrist (0.5, 0.90) to middle-finger base (0.5, 0.58)
HAND_SIZE = 0.32


def make_hand(index=False, middle=False, ring=False, pinky=False,
              thumb="neutral"):
    """
    Build 21 landmarks for an upright hand.

    index/middle/ring/pinky: True = finger extended, False = curled.
    thumb: "up", "down" or "neutral" (tucked across the palm).
    """
    points = [(0.5, 0.5)] * 21

    points = list(points)
    points[0] = (0.5, 0.90)  # wrist

    raised = {
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

    for name, (mcp, pip, dip, tip) in FINGERS.items():
        x = FINGER_X[name]
        points[mcp] = (x, 0.60 if name != "middle" else 0.58)
        points[pip] = (x, 0.50)

        if raised[name]:
            points[dip] = (x, 0.40)
            points[tip] = (x, 0.30)  # tip above PIP -> finger up
        else:
            points[dip] = (x, 0.58)
            points[tip] = (x, 0.62)  # tip below PIP -> finger curled

    points[1] = (0.40, 0.82)
    points[2] = (0.36, 0.75)  # thumb base (MCP)

    # The thumb is judged relative to the index-finger base (landmark 5,
    # y = 0.60) in units of hand size (wrist -> middle base = 0.32).
    # thumb_ratio > 0 means the tip is above the index base.
    if isinstance(thumb, (int, float)):
        thumb_ratio = float(thumb)
    elif thumb == "up":
        thumb_ratio = 0.47
    elif thumb == "down":
        thumb_ratio = -0.78
    else:
        thumb_ratio = -0.06  # tucked into a fist

    tip_y = 0.60 - thumb_ratio * HAND_SIZE
    points[3] = (0.35, (points[2][1] + tip_y) / 2)
    points[4] = (0.34, tip_y)

    return points


# ==========================================
# detect_gesture
# ==========================================

def test_open_palm():
    hand = make_hand(index=True, middle=True, ring=True, pinky=True)
    assert detect_gesture(hand) == "OPEN PALM"


def test_fist():
    hand = make_hand(thumb="neutral")
    assert detect_gesture(hand) == "FIST"


def test_thumbs_up():
    hand = make_hand(thumb="up")
    assert detect_gesture(hand) == "THUMBS UP"


def test_thumbs_down():
    hand = make_hand(thumb="down")
    assert detect_gesture(hand) == "THUMBS DOWN"


def test_two_fingers():
    hand = make_hand(index=True, middle=True)
    assert detect_gesture(hand) == "TWO FINGERS"


def test_one_finger():
    hand = make_hand(index=True)
    assert detect_gesture(hand) == "ONE FINGER"


def test_unknown_for_unmapped_pose():
    # Ring + pinky only is not a supported gesture
    hand = make_hand(ring=True, pinky=True)
    assert detect_gesture(hand) == "UNKNOWN"

    # Three fingers is not a supported gesture either
    hand = make_hand(index=True, middle=True, ring=True)
    assert detect_gesture(hand) == "UNKNOWN"


def test_fist_is_not_confused_with_thumbs():
    # A closed fist has the thumb across the fingers, not pointing up/down
    assert detect_gesture(make_hand(thumb="neutral")) not in (
        "THUMBS UP", "THUMBS DOWN"
    )

    # Even with the thumb resting high on the fist it is still a fist
    assert detect_gesture(make_hand(thumb=0.15)) == "FIST"
    assert detect_gesture(make_hand(thumb=-0.15)) == "FIST"


def test_modest_thumb_extension_still_counts():
    # Real thumbs-ups are not as extreme as the textbook pose
    assert detect_gesture(make_hand(thumb=0.40)) == "THUMBS UP"
    assert detect_gesture(make_hand(thumb=-0.40)) == "THUMBS DOWN"


def test_hand_metrics():
    fingers, ratio = hand_metrics(make_hand(index=True, middle=True))
    assert fingers == 2

    _, ratio = hand_metrics(make_hand(thumb=0.47))
    assert math.isclose(ratio, 0.47, abs_tol=1e-6)

    _, ratio = hand_metrics(make_hand(thumb=-0.78))
    assert math.isclose(ratio, -0.78, abs_tol=1e-6)


def test_gesture_is_independent_of_hand_scale():
    # Scaling every landmark (hand closer/further from the camera) must not
    # change the result, because the thumb threshold is relative to hand size
    for kwargs, expected in [
        ({"thumb": "up"}, "THUMBS UP"),
        ({"thumb": "down"}, "THUMBS DOWN"),
        ({"thumb": "neutral"}, "FIST"),
    ]:
        hand = make_hand(**kwargs)
        scaled = [(x * 640, y * 480) for x, y in hand]
        assert detect_gesture(scaled) == expected


def test_invalid_landmarks_return_unknown():
    assert detect_gesture(None) == "UNKNOWN"
    assert detect_gesture([]) == "UNKNOWN"
    assert detect_gesture([(0.5, 0.5)] * 5) == "UNKNOWN"


def test_distance():
    assert distance((0, 0), (3, 4)) == 5
    assert math.isclose(distance((1, 1), (1, 1)), 0)


# ==========================================
# Cooldown
# ==========================================

def test_cooldown_ready():
    assert cooldown_ready(0.0, 10.0, 1.0)
    assert cooldown_ready(10.0, 11.0, 1.0)
    assert not cooldown_ready(10.0, 10.5, 1.0)


def test_cooldown_gestures_set():
    assert COOLDOWN_GESTURES == {"THUMBS UP", "THUMBS DOWN", "TWO FINGERS"}
    # Screenshot fires once per hold; volume may repeat
    assert ONE_SHOT_GESTURES == {"TWO FINGERS"}


# ==========================================
# GestureStabilizer
# ==========================================

def test_stabilizer_needs_consecutive_frames():
    stabilizer = GestureStabilizer(required_frames=3)

    assert stabilizer.update("THUMBS UP") == ("NO HAND", False)
    assert stabilizer.update("THUMBS UP") == ("NO HAND", False)
    assert stabilizer.update("THUMBS UP") == ("THUMBS UP", True)
    # Still held: stays stable but is not a new change
    assert stabilizer.update("THUMBS UP") == ("THUMBS UP", False)


def test_stabilizer_ignores_single_frame_flicker():
    stabilizer = GestureStabilizer(required_frames=3)

    for _ in range(3):
        stabilizer.update("OPEN PALM")

    # One misread frame must not change the stable gesture
    assert stabilizer.update("TWO FINGERS") == ("OPEN PALM", False)
    assert stabilizer.update("OPEN PALM") == ("OPEN PALM", False)


def test_stabilizer_fires_again_after_release():
    stabilizer = GestureStabilizer(required_frames=2)

    stabilizer.update("TWO FINGERS")
    assert stabilizer.update("TWO FINGERS") == ("TWO FINGERS", True)

    stabilizer.update("FIST")
    assert stabilizer.update("FIST") == ("FIST", True)

    stabilizer.update("TWO FINGERS")
    assert stabilizer.update("TWO FINGERS") == ("TWO FINGERS", True)


# ==========================================
# Mouse mapping
# ==========================================

def test_map_to_screen_corners_and_center():
    w, h = 1920, 1080

    assert map_to_screen(0.5, 0.5, w, h, margin=0.15) == (
        (w - 1) / 2, (h - 1) / 2
    )
    assert map_to_screen(0.15, 0.15, w, h, margin=0.15) == (0, 0)
    assert map_to_screen(0.85, 0.85, w, h, margin=0.15) == (w - 1, h - 1)


def test_map_to_screen_clamps_outside_margin():
    w, h = 1000, 500

    assert map_to_screen(0.0, 0.0, w, h, margin=0.15) == (0, 0)
    assert map_to_screen(1.0, 1.0, w, h, margin=0.15) == (w - 1, h - 1)


def test_smooth_position():
    assert smooth_position(None, (100, 200)) == (100, 200)

    x, y = smooth_position((0, 0), (100, 100), smoothing=0.5)
    assert (x, y) == (50, 50)

    # smoothing = 0 means jump straight to the target
    assert smooth_position((0, 0), (10, 20), smoothing=0) == (10, 20)


# ==========================================
# perform_action
# ==========================================

def test_perform_action_volume_up():
    with mock.patch.object(gesture_actions, "volume_up") as volume_up:
        assert perform_action("THUMBS UP", None, 0) == "Volume Up"
        volume_up.assert_called_once()


def test_perform_action_volume_down():
    with mock.patch.object(gesture_actions, "volume_down") as volume_down:
        assert perform_action("THUMBS DOWN", None, 0) == "Volume Down"
        volume_down.assert_called_once()


def test_perform_action_screenshot_writes_file():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    original_dir = os.getcwd()

    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            assert perform_action("TWO FINGERS", frame, 1234.5) == "Screenshot"
            assert os.path.exists("screenshot_1234.png")
        finally:
            os.chdir(original_dir)


def test_perform_action_labels():
    assert perform_action("FIST", None, 0) == "Fist Detected"
    assert perform_action("ONE FINGER", None, 0) == "Mouse Control"
    assert perform_action("OPEN PALM", None, 0) == "Open Palm"
    assert perform_action("NO HAND", None, 0) == "Waiting..."
    assert perform_action("UNKNOWN", None, 0) == "UNKNOWN"


def test_volume_does_not_crash_when_pactl_missing():
    with mock.patch.object(
        gesture_actions.subprocess, "run", side_effect=FileNotFoundError
    ), mock.patch.object(gesture_actions, "_get_pyautogui", return_value=None):
        gesture_actions.volume_up()
        gesture_actions.volume_down()


if __name__ == "__main__":
    tests = [
        (name, func) for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func)
    ]

    for name, func in tests:
        func()
        print(f"PASS  {name}")

    print(f"\nAll {len(tests)} tests passed!")
