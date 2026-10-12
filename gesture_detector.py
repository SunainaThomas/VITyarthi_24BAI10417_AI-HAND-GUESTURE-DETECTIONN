import math

from config import STABLE_FRAMES, THUMB_EXTENSION_RATIO


def distance(point1, point2):
    """Calculate Euclidean distance between two landmarks."""
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def hand_metrics(landmarks):
    """
    Return the number of extended fingers and normalized thumb direction.

    Landmarks must contain 21 (x, y) points.
    Coordinates can be normalized or pixel coordinates.
    """

    if landmarks is None or len(landmarks) < 21:
        return 0, 0.0

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    hand_size = distance(wrist, middle_mcp)

    if hand_size < 1e-6:
        return 0, 0.0

    # Detect extended fingers using distances from the wrist.
    # This is more reliable than comparing vertical positions alone.
    fingers = 0

    for tip_index, pip_index in [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18),  # Pinky
    ]:
        tip_distance = distance(landmarks[tip_index], wrist)
        pip_distance = distance(landmarks[pip_index], wrist)

        if tip_distance > pip_distance * 1.10:
            fingers += 1

    # Positive = thumb above index base.
    # Negative = thumb below index base.
    thumb_ratio = (
        landmarks[5][1] - landmarks[4][1]
    ) / hand_size

    return fingers, thumb_ratio


def detect_gesture(landmarks):
    """
    Recognize:
    OPEN PALM, FIST, THUMBS UP, THUMBS DOWN,
    TWO FINGERS, ONE FINGER, UNKNOWN.

    Best results occur when the palm faces the camera
    and the hand is approximately upright.
    """

    if landmarks is None or len(landmarks) < 21:
        return "UNKNOWN"

    fingers, thumb_ratio = hand_metrics(landmarks)

    index_up = landmarks[8][1] < landmarks[6][1]
    middle_up = landmarks[12][1] < landmarks[10][1]
    ring_up = landmarks[16][1] < landmarks[14][1]
    pinky_up = landmarks[20][1] < landmarks[18][1]

    # 1. Open palm
    if fingers == 4:
        return "OPEN PALM"

    # 2. Two fingers
    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):
        return "TWO FINGERS"

    # 3. One finger
    if (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "ONE FINGER"

    # 4. Thumb gestures when the other fingers are folded
    if fingers == 0:

        # Relaxed threshold for clearer thumb detection.
        threshold = max(0.12, THUMB_EXTENSION_RATIO)

        if thumb_ratio > threshold:
            return "THUMBS UP"

        if thumb_ratio < -threshold:
            return "THUMBS DOWN"

        return "FIST"

    return "UNKNOWN"


class GestureStabilizer:
    """
    Accept a gesture only after it appears for several consecutive frames.
    Returns (stable_gesture, changed).
    """

    def __init__(self, required_frames=STABLE_FRAMES):
        self.required_frames = max(1, required_frames)
        self.stable = "NO HAND"
        self._candidate = None
        self._count = 0

    def update(self, gesture):

        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1

        changed = False

        if (
            self._count >= self.required_frames
            and gesture != self.stable
        ):
            self.stable = gesture
            changed = True

        return self.stable, changed
