"""Configuration values for the AI Hand Gesture Controller."""

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe confidence thresholds
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6

# Minimum seconds between two triggered actions (volume, screenshot)
ACTION_COOLDOWN = 1.0

# A gesture must be seen on this many frames in a row before it counts.
# Stops a single misread frame from firing an action.
STABLE_FRAMES = 4

# Thumb must point up/down past the index-finger base by at least this
# fraction of the hand size to count as THUMBS UP / THUMBS DOWN.
# Lower = easier to trigger, higher = stricter. Press 'd' while running to see
# the live value ("thumb") and tune this.
THUMB_EXTENSION_RATIO = 0.35

# Mouse control (ONE FINGER gesture)
# Fraction of the frame, on each side, ignored so the cursor can reach
# the screen edges without the hand leaving the camera view.
MOUSE_FRAME_MARGIN = 0.15
# 0 = no smoothing, closer to 1 = smoother but laggier cursor
MOUSE_SMOOTHING = 0.6

WINDOW_NAME = "AI Hand Gesture Controller"
QUIT_KEY = "q"
DEBUG_KEY = "d"  # toggles the debug line (finger count + thumb value)

GESTURES = [
    "THUMBS UP",
    "THUMBS DOWN",
    "TWO FINGERS",
    "FIST",
    "ONE FINGER",
    "OPEN PALM",
]
