import subprocess

import cv2

from config import MOUSE_FRAME_MARGIN, MOUSE_SMOOTHING

# Gestures that trigger an action and are rate-limited by the cooldown in
# main.py. (Mouse control runs every frame, so it is not in this set.)
COOLDOWN_GESTURES = {"THUMBS UP", "THUMBS DOWN", "TWO FINGERS"}

# Of those, gestures that fire only once per hold. Volume repeats every
# cooldown while the thumb is held (like holding a volume key), but holding
# two fingers must take ONE screenshot, not one every second.
ONE_SHOT_GESTURES = {"TWO FINGERS"}


# ==========================================
# Cooldown
# ==========================================

def cooldown_ready(last_action_time, current_time, cooldown):
    """True if at least `cooldown` seconds passed since the last action."""
    return (current_time - last_action_time) >= cooldown


# ==========================================
# System volume
# ==========================================

def _change_volume(percent_change, fallback_key):
    """
    Change the volume with PulseAudio/PipeWire (`pactl`) on Linux.
    If `pactl` is not installed (Windows/macOS), press the media key via
    PyAutoGUI instead. Never raises, so a missing tool cannot crash the loop.
    """
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", percent_change],
            check=False
        )
        return
    except FileNotFoundError:
        pass

    pyautogui = _get_pyautogui()

    if pyautogui is not None:
        try:
            pyautogui.press(fallback_key)
        except Exception as error:
            print(f"Volume control failed: {error}")
    else:
        print("Volume control unavailable: install pactl or pyautogui.")


def volume_up():
    _change_volume("+5%", "volumeup")


def volume_down():
    _change_volume("-5%", "volumedown")


# ==========================================
# Screenshot
# ==========================================

def take_screenshot(frame, timestamp):
    filename = f"screenshot_{int(timestamp)}.png"
    cv2.imwrite(filename, frame)

    print(f"Screenshot saved: {filename}")

    return filename


# ==========================================
# Mouse control (ONE FINGER gesture)
# ==========================================

_pyautogui = None
_pyautogui_failed = False
_previous_mouse = None  # last smoothed cursor position (x, y)


def _get_pyautogui():
    """
    Import PyAutoGUI lazily. It needs a display, so importing it at module
    level would break tests and headless machines.
    """
    global _pyautogui, _pyautogui_failed

    if _pyautogui is None and not _pyautogui_failed:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            _pyautogui = pyautogui
        except Exception as error:
            _pyautogui_failed = True
            print(f"PyAutoGUI unavailable, mouse control disabled: {error}")

    return _pyautogui


def map_to_screen(x_norm, y_norm, screen_width, screen_height,
                  margin=MOUSE_FRAME_MARGIN):
    """
    Map a normalised (0-1) fingertip position in the camera frame to screen
    pixels. A margin around the frame is ignored so the cursor can reach the
    screen edges without the hand leaving the camera view.
    """
    usable = 1 - 2 * margin

    x = (x_norm - margin) / usable
    y = (y_norm - margin) / usable

    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)

    return x * (screen_width - 1), y * (screen_height - 1)


def smooth_position(previous, target, smoothing=MOUSE_SMOOTHING):
    """Exponential smoothing so the cursor does not jitter."""
    if previous is None:
        return target

    return (
        previous[0] * smoothing + target[0] * (1 - smoothing),
        previous[1] * smoothing + target[1] * (1 - smoothing),
    )


def control_mouse(x_norm, y_norm):
    """Move the cursor to follow the index fingertip. Returns True if moved."""
    global _previous_mouse

    pyautogui = _get_pyautogui()

    if pyautogui is None:
        return False

    screen_width, screen_height = pyautogui.size()

    target = map_to_screen(x_norm, y_norm, screen_width, screen_height)
    _previous_mouse = smooth_position(_previous_mouse, target)

    pyautogui.moveTo(int(_previous_mouse[0]), int(_previous_mouse[1]))

    return True


def reset_mouse():
    """Forget the last cursor position (call when the hand is lost)."""
    global _previous_mouse
    _previous_mouse = None


# ==========================================
# Gesture -> action mapping
# ==========================================

def perform_action(gesture, frame, current_time):
    if gesture == "THUMBS UP":
        volume_up()
        return "Volume Up"

    elif gesture == "THUMBS DOWN":
        volume_down()
        return "Volume Down"

    elif gesture == "TWO FINGERS":
        take_screenshot(frame, current_time)
        return "Screenshot"

    elif gesture == "FIST":
        return "Fist Detected"

    elif gesture == "ONE FINGER":
        # The cursor itself is moved every frame by control_mouse() in main.py
        return "Mouse Control"

    elif gesture == "OPEN PALM":
        return "Open Palm"

    elif gesture == "NO HAND":
        return "Waiting..."

    return gesture
