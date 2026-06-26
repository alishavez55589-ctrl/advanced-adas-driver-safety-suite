# Advanced ADAS Safety Suite Dashboard

An industry-level, production-ready Advanced Driver Assistance System (ADAS) built using Python, OpenCV, and the modern MediaPipe Tasks API. This system monitors driver behavior in real-time to prevent accidents caused by fatigue, distractions, or safety violations.

## Key Features
* Fatigue/Drowsiness Parsing: Uses Eye Aspect Ratio (EAR) with a strict 5-second validation matrix to trigger alerts.
* Yawn Detection: Monitors Mouth Aspect Ratio (MAR) to evaluate driver tiredness.
* Spatial Head Orientation: Tracks absolute distractions and mobile usage via dynamic head alignment vector estimations.
* Safety Restraint Validation: Posture and seatbelt strap checking using body pose landmarks.
* Zero-Dependency Audio Alarm Engine: Uses host OS-level hardware beeps to alarm drivers asynchronously without causing loop bottlenecks.
* Dynamic Resolution UI Overlay: Telemetry panel scales perfectly across different camera aspect ratios (Defaulted to HD 1280x720 standard).

## Tech Stack
* OpenCV 4.13.0.92
* MediaPipe 0.10.35 (Modern Tasks API Standard)
* Python (Multi-threaded execution)

## Installation & Setup

1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Install dependencies:
pip install -r requirements.txt

3. Download Model Assets (MediaPipe Tasks API requires localized .task bundles):
Place face_landmarker.task and pose_landmarker_full.task into the root directory.

4. Run the Pipeline:
python main.py