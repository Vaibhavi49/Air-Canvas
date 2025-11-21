🎨 Air Canvas

A gesture-controlled drawing application where you draw in the air using just your hand. Built with OpenCV and fingertip detection, Air Canvas turns your camera feed into a virtual whiteboard that reacts to your movements in real time.

🚀 Overview

Air Canvas tracks your hand through the webcam, identifies your index fingertip, and uses its coordinates to draw on the screen. With simple gestures, you can switch between colors, draw, erase, and even clear the screen—no physical stylus needed.

This project is perfect for learning computer vision, gesture recognition, and real-time image processing.

✨ Features

🖐️ Hand Detection using OpenCV + MediaPipe

📍 Fingertip Tracking for accurate drawing

🎨 Multiple Colors (Red, Blue, Green, etc.)

🧽 Eraser Mode

🖥️ Clean UI for color selection

⚡ Real-Time Rendering

👩‍💻 Beginner-friendly and fully customizable

🧠 How It Works

The webcam captures frames.

MediaPipe detects hand landmarks.

Index fingertip (landmark 8) is tracked.

When the finger is up, lines are drawn between consecutive fingertip points.

If fingertip overlaps with color boxes → color changes.

If fingertip enters eraser zone → it erases drawn strokes.

A mask layer is used to ensure smooth drawing without flickering.

🛠️ Tech Stack

Python

OpenCV

MediaPipe

NumPy
