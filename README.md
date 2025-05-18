# ResQon Drone Project

드론을 활용한 인명 탐지 및 비상 상황 대응 프로젝트입니다.

## 프로젝트 구조

```
projectDrone/
├── yolov5/                  # 학습된 YOLOv5 모델 및 관련 코드
├── models/                  # 학습된 모델 파일들
├── emergency_detecter.py    # 비상 상황 감지 모듈
├── drone_detection.py       # 드론 객체 감지 모듈
├── pathFinding.py          # 경로 탐색 알고리즘
└── upgradeDrone_detection.py # 개선된 드론 감지 모듈
```

## 설치 방법

1. Python 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 드론 객체 감지:
```bash
python drone_detection.py
```

2. 비상 상황 감지:
```bash
python emergency_detecter.py
```

## 주의사항

- YOLOv5 모델 파일(*.pt)은 용량이 크므로 .gitignore에 포함되어 있습니다.
- 모델 파일은 별도로 다운로드하거나 학습된 모델을 사용해야 합니다. 