# ECHO HANDS — Real-Time Sign Language Recognition & Speech

## Project Overview & Pitch

ECHO HANDS is an intelligent real-time sign language recognition system designed to bridge the communication gap between sign language users and non-signers. The project uses computer vision, hand tracking, and speech synthesis to detect live hand gestures from a webcam, recognize ASL letters, build readable words, and convert them into spoken English in real time.

The goal is to make communication more inclusive, accessible, and natural for deaf and hard-of-hearing users. Instead of relying on manual transcription or static gesture recognition, ECHO HANDS continuously interprets live hand movements, stabilizes uncertain predictions, and speaks the recognized word aloud for clear, real-time communication.

This system is useful for everyday communication support, educational tools, accessibility solutions, and inclusive AI applications. It demonstrates how modern computer vision and speech technology can break down communication barriers and create meaningful, human-centered experiences.

---

## What the Project Does

ECHO HANDS performs the following tasks:

- Detects a hand from a live camera feed
- Tracks the hand using MediaPipe landmarks
- Recognizes ASL gestures from A to Z
- Detects dynamic motion-based letters such as J and Z
- Stabilizes predictions to avoid jitter and false fluctuations
- Builds words from recognized letters
- Speaks the final word using natural browser-based speech synthesis
- Offers replay, mute, and clear controls for better interaction
- Displays real-time status, confidence, and word output in a user-friendly dashboard

---

## Tech Stack & System Integration

The project combines multiple technologies to create a working end-to-end assistive communication system:

- Python — main application language
- OpenCV — webcam capture and real-time frame processing
- MediaPipe — hand landmark detection and tracking
- NumPy — mathematical operations for finger geometry and coordinate processing
- Streamlit — browser-based user interface and live app interaction
- Web Speech API — browser-native voice output for natural pronunciation
- pyttsx3 — desktop speech fallback for local or terminal execution

These components work together in a connected pipeline:

1. The webcam captures live frames using OpenCV.
2. MediaPipe detects and tracks 21 hand landmarks.
3. NumPy computes finger geometry, distances, and orientation features.
4. The recognition engine matches these features against ASL letter patterns.
5. The stabilizer filters uncertain predictions and removes flashing false positives.
6. The word builder accumulates confirmed letters and forms readable words.
7. The speech engine converts the final recognized word to spoken output.
8. Streamlit displays the live camera feed, status, and interactive controls.

---

## File Structure & How It Connects

- [app.py](app.py) — main Streamlit application; manages the user interface, camera loop, recognition flow, and speech output integration
- [hand_detector.py](hand_detector.py) — detects hands and extracts landmark coordinates from camera frames
- [recognizer.py](recognizer.py) — core recognition logic for ASL letters, dynamic gestures, geometry analysis, and confidence scoring
- [word_builder.py](word_builder.py) — accumulates recognized letters into words and handles hold-to-confirm behavior
- [tts.py](tts.py) — manages speech generation, browser audio output, and desktop voice fallback
- [main.py](main.py) — alternative OpenCV-based desktop runner for non-Streamlit execution
- [requirements.txt](requirements.txt) — required Python libraries for the project

These modules are connected through a single real-time pipeline: the camera -> hand detector -> recognizer -> word builder -> speech engine -> user interface.

---

## Complete 26-Letter ASL Recognition (A–Z) & Natural Text-to-Speech

ECHO HANDS is an AI-powered sign language recognition application that uses a camera to recognize hand gestures across the entire 26-letter English alphabet using American Sign Language (ASL), construct readable words, and pronounce them aloud using natural text-to-speech.

---

## Running the Application

Ensure the local virtual environment is active before starting the app.

### Option A: Streamlit Web Application (Primary)

```bash
./run.sh
```

Or:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Option B: OpenCV Desktop Application

```bash
python main.py
```

Or:

```bash
.venv/bin/python main.py
```

---

## Key Capabilities

1. Complete Alphabet Support (A to Z)
2. Dynamic motion gestures for J and Z
3. Accurate hand tracking with MediaPipe
4. Real-time word construction
5. Natural speech synthesis
6. Replay and mute controls
7. Anti-repetition protection and stabilization

---

## Project Summary

ECHO HANDS is a real-time ASL recognition and speech synthesis system that turns live hand gestures into spoken English. By combining computer vision, gesture recognition, and assistive AI, the project helps reduce communication barriers and improve accessibility for sign language users.
