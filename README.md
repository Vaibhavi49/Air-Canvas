Air Canvas

Draw in the air using hand gestures. This project uses OpenCV and fingertip tracking to let you choose colors, draw, and erase without touching the screen—your hand becomes the pen.

Features

Real-time hand + fingertip detection

Draw with multiple colors

Eraser mode

Smooth drawing experience

Simple and beginner-friendly code

Tech Used

Python

OpenCV

MediaPipe (or custom detection)

How It Works

The camera tracks your hand, detects your index fingertip, and maps its movement onto the screen. Gestures control drawing, color selection, and erasing.

Setup
pip install opencv-python mediapipe numpy

Run
python air_canvas.py

Use Cases

Virtual whiteboard, gesture-based UI, computer vision learning.
