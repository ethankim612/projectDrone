# ResQon Drone Project

A project for human detection and emergency response using drones.

## Project Structure

```
projectDrone/
├── yolov5/                  # Trained YOLOv5 models and related code
├── models/                  # Trained model files
├── emergency_detecter.py    # Emergency situation detection module
├── drone_detection.py       # Drone object detection module
├── pathFinding.py           # Pathfinding algorithm
└── upgradeDrone_detection.py # Improved drone detection module
```

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Drone object detection:
```bash
python drone_detection.py
```

2. Emergency situation detection:
```bash
python emergency_detecter.py
```

## Notes

- YOLOv5 model files (*.pt) are excluded from git via .gitignore due to their large size.
- You need to download or train the model files separately. 