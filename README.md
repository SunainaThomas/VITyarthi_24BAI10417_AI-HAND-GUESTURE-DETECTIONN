# AI Hand Gesture Controller

A real-time computer vision project that lets you control computer functions with hand gestures through a webcam.

## Features

| Gesture | Action |
|---|---|
| 👍 Thumbs Up | Increase system volume |
| 👎 Thumbs Down | Decrease system volume |
| ✌️ Two Fingers | Take a screenshot (`screenshot_<time>.png`) |
| ☝️ One Finger | Mouse control: the cursor follows your index fingertip |
| ✊ Fist | Detected, no action |
| 🖐️ Open Palm | Neutral / stop |

Also included:

- Real-time hand landmark detection (MediaPipe)
- Live on-screen display of gesture, action and FPS
- Gesture stabilizer: a gesture must stay steady for a few frames before it counts, so one misread frame does not trigger an action
- Cooldown (default 1 second) between actions; holding Thumbs Up/Down repeats the volume change every cooldown, while Two Fingers takes only one screenshot per hold (lower your hand to take another)
- Cursor smoothing and edge margin for stable mouse control
- Camera check with a clear error message, and clean exit with `q`
- Press `d` while running to show live debug info (raw gesture, finger count, thumb value)

## Technologies Used

- Python 3.9 - 3.12
- OpenCV
- MediaPipe (0.10.14, uses the `mp.solutions` API)
- NumPy
- PyAutoGUI (mouse control, and volume keys on Windows/macOS)
- Linux PulseAudio/PipeWire (`pactl`) for volume on Linux
- pytest

## How It Works

The webcam captures live video frames. Each frame is mirrored and converted from BGR to RGB, and MediaPipe detects the hand and extracts 21 landmarks. The gesture module compares landmark positions (for example, whether a fingertip is above its middle joint) to identify the gesture. The gesture is then mapped to a computer action.

```text
Webcam
   ↓
OpenCV (flip, BGR → RGB)
   ↓
MediaPipe Hand Detection
   ↓
Hand Landmarks (21 points)
   ↓
Gesture Recognition (gesture_detector.py)
   ↓
Cooldown check + Computer Action (gesture_actions.py)
```

## Project Structure

```text
main.py              Webcam loop: capture, detect, act, display, quit
gesture_detector.py  detect_gesture(landmarks) -> gesture name
gesture_actions.py   Volume, screenshot, mouse control, cooldown helper
utils.py             FPS calculation and on-screen info panel
config.py            Camera, confidence, cooldown and mouse settings
test_gestures.py     Unit tests using synthetic landmark data
```

## Installation

```bash
git clone https://github.com/SunainaThomas/VITyarthi_24BAI10417_AI-HAND-GUESTURE-DETECTION.git
cd VITyarthi_24BAI10417_AI-HAND-GUESTURE-DETECTION

python -m venv venv
# Linux/macOS:  source venv/bin/activate
# Windows:      venv\Scripts\activate

pip install -r requirements.txt
```

Use Python 3.9 - 3.12. Newer MediaPipe releases removed the `mp.solutions` API this project uses, which is why `requirements.txt` pins `mediapipe==0.10.14`.

## Usage

```bash
python main.py
```

Show your hand to the camera, upright and facing it, in good light. Press `q` in the video window to quit.

Settings such as camera index, cooldown, mouse smoothing and gesture sensitivity are in `config.py`.

Tips for reliable detection:

- Keep your hand upright and fully in view, about an arm's length from the camera.
- For Thumbs Up/Down, curl the other four fingers into a fist and point the thumb clearly up or down.
- Open Palm is a neutral gesture, so it does not trigger an action.
- If Thumbs Up/Down is not detected, press `d` and watch the `thumb` value while making the gesture. Lower `THUMB_EXTENSION_RATIO` in `config.py` if the value stays below it, or raise it if a fist is read as a thumbs gesture.

## Running the Tests

```bash
pytest -v
# or
python test_gestures.py
```

The tests build synthetic MediaPipe-style landmarks and check that `detect_gesture` returns the right gesture for every supported pose (and `UNKNOWN` for unsupported ones), plus the cooldown, mouse mapping, smoothing and action mapping. No webcam is required.

## Limitations

- Gestures assume an upright hand facing the camera; a hand rotated sideways or upside down may be misread.
- Only one hand is tracked.
- On Linux, volume control needs `pactl`; on Windows/macOS it falls back to PyAutoGUI volume keys.
- Mouse control needs a desktop display (it will not work over SSH or headless).
