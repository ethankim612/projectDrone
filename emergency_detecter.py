import cv2
import time
import os
import numpy as np
from datetime import datetime
from threading import Thread
import queue
import math
import requests
import json
from ultralytics import YOLO  # YOLOv8 사용
import urllib.parse

class EmergencyDetector:
    def __init__(self, rtmp_url, api_endpoint=None):
        # 기본 설정
        self.rtmp_url = rtmp_url
        self.frame_queue = queue.Queue(maxsize=1)  # 프레임 큐 크기를 1로 줄임
        self.stopped = False
        
        # API 엔드포인트 (소방청 API 등)
        self.api_endpoint = api_endpoint
        self.last_api_call_time = 0
        self.api_call_interval = 5  # 5초마다 API 호출 (조정 가능)
        
        # 저장 관련 설정
        self.save_dir = "emergency_detections"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # models 디렉토리 확인 및 생성
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # age_deploy.prototxt 파일 생성 (없는 경우)
        age_proto_path = os.path.join(models_dir, "age_deploy.prototxt")
        if not os.path.exists(age_proto_path):
            self.create_age_proto_file(age_proto_path)
            print(f"age_deploy.prototxt 파일이 생성되었습니다: {age_proto_path}")
        
        # 로그 파일 설정
        self.log_file = os.path.join(self.save_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, "w") as f:
            f.write("timestamp,person_id,bbox_x,bbox_y,bbox_width,bbox_height,estimated_age,injury_probability,posture,sent_to_api\n")
        
        # 녹화 관련 설정
        self.recording = False
        self.video_writer = None
        
        # 감지 관련 설정
        self.detection_enabled = True
        self.detection_confidence = 0.35
        self.process_every_n_frames = 2
        self.counter = 0
        
        # 디스플레이 설정
        self.display_width = 1280
        self.display_height = 720
        
        # 화면 관련 변수
        self.processed_frame = None
        
        # 거리 추정 관련 변수
        self.default_object_sizes = {
            'person': 1.7,  # 평균 사람 키 (미터)
            'car': 1.5,
            'truck': 2.5,
            'motorcycle': 1.2,
            'bicycle': 1.0,
            'drone': 0.3,
            'default': 0.5
        }
        self.focal_length = None
        self.camera_fov = 75.0
        
        # 색상 맵 (클래스별 색상)
        self.color_map = {}
        
        # 사람 추적 관련 변수
        self.tracked_persons = {}  # 추적 ID를 키로 사용
        self.next_person_id = 1
        self.person_tracking_timeout = 30  # 30프레임 동안 안 보이면 추적 중단
        
        # 나이 추정 모델
        self.age_model = None
        self.face_detector = None
        
        # 자세 분석 모델
        self.pose_model = None
        
        # 부상 탐지 관련 변수
        self.injury_thresholds = {
            'falling': 0.7,     # 쓰러짐 감지 임계값
            'unusual_pose': 0.6,  # 비정상 자세 임계값
            'not_moving': 0.5   # 움직임 없음 임계값
        }
        
        # 나이 그룹 정의
        self.age_groups = {
            (0, 12): "어린이",
            (13, 19): "청소년",
            (20, 65): "성인",
            (66, 100): "노인"
        }

    def create_age_proto_file(self, file_path):
        """age_deploy.prototxt 파일 생성"""
        proto_content = """name: "AgeSolver"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 227
      dim: 227
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  convolution_param {
    num_output: 256
    kernel_size: 5
    stride: 1
    pad: 2
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  convolution_param {
    num_output: 384
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc4"
  type: "InnerProduct"
  bottom: "pool3"
  top: "fc4"
  inner_product_param {
    num_output: 512
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "fc4"
  top: "fc4"
}
layer {
  name: "drop4"
  type: "Dropout"
  bottom: "fc4"
  top: "fc4"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc5"
  type: "InnerProduct"
  bottom: "fc4"
  top: "fc5"
  inner_product_param {
    num_output: 512
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "fc5"
  top: "fc5"
}
layer {
  name: "drop5"
  type: "Dropout"
  bottom: "fc5"
  top: "fc5"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc6-age"
  type: "InnerProduct"
  bottom: "fc5"
  top: "fc6-age"
  inner_product_param {
    num_output: 8
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc6-age"
  top: "prob"
}"""
        
        # 파일 쓰기
        with open(file_path, 'w') as f:
            f.write(proto_content)

    def start_capture(self):
        """RTMP 스트림 캡처 시작"""
        print(f"RTMP 스트림에 연결 시도 중: {self.rtmp_url}")
        
        # FFMPEG 설정 - 극단적인 저지연 설정
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;0|max_delay;0|fflags;discardcorrupt+nobuffer+fastseek|flags;low_delay"
        
        # 캡처 객체 생성
        self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        # 연결 확인
        if not self.cap.isOpened():
            print("Error: RTMP 스트림을 열 수 없습니다.")
            return False
        
        # 프레임 정보 가져오기
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"RTMP 스트림 연결 성공! (초저지연 모드)")
        print(f"영상 정보: {self.width}x{self.height}, {self.fps}fps")
        
        # 초점 거리 계산 (카메라 캘리브레이션)
        self.focal_length = (self.width * 0.5) / math.tan(math.radians(self.camera_fov * 0.5))
        print(f"계산된 초점 거리: {self.focal_length:.2f} 픽셀")
        
        # 캡처 스레드 시작
        self.capture_thread = Thread(target=self._capture_frames, name="CaptureThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def load_models(self):
        """모든 필요한 모델 로드"""
        print("모델 로드 중...")
        
        try:
            # 1. YOLOv8 모델 로드 (객체 탐지)
            self.model = YOLO("yolov8n.pt")
            
            # 모델 클래스 목록 가져오기
            self.classes = self.model.names
            
            # 클래스별 랜덤 색상 생성
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
            
            # 색상 맵 생성
            for i, class_name in self.classes.items():
                self.color_map[class_name] = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            
            # 2. 얼굴 감지 모델 로드
            print("얼굴 감지 모델 로드 중...")
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 3. 나이 추정 모델 로드 (가능한 모델)
            print("나이 추정 모델 로드 중...")
            try:
                # 모델 파일 경로
                age_proto = "models/age_deploy.prototxt"
                age_model = "models/age_net.caffemodel"
                
                # 모델 파일이 없으면 다운로드 안내
                if not os.path.exists(age_model):
                    print("age_net.caffemodel 파일이 필요합니다. 다음 URL에서 다운로드 가능합니다:")
                    print("1. https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel")
                    print("2. https://drive.google.com/file/d/1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW/view?usp=sharing")
                    print("다운로드 후 'models' 폴더에 저장하세요.")
                    print("지금은 간단한 추정값을 사용합니다.")
                else:
                    self.age_model = cv2.dnn.readNet(age_model, age_proto)
                    print("나이 추정 모델 로드 성공!")
            except Exception as e:
                print(f"나이 추정 모델 로드 오류: {e}")
                print("간단한 추정값을 사용합니다.")
            
            # 4. 자세 분석 모델 (YOLOv8의 pose 모델 사용)
            try:
                print("자세 추정 모델 로드 중...")
                self.pose_model = YOLO("yolov8n-pose.pt")
                print("자세 추정 모델 로드 성공!")
            except Exception as e:
                print(f"자세 추정 모델 로드 오류: {e}")
                print("자세 추정은 생략됩니다.")
            
            print("모든 모델 로드 완료")
            return True
        except Exception as e:
            print(f"모델 로드 총괄 오류: {e}")
            return False
    
    def _capture_frames(self):
        """프레임 캡처 스레드 함수 (저지연 최적화)"""
        while not self.stopped:
            if not self.cap.isOpened():
                print("캡처 연결이 끊어졌습니다. 재연결 시도...")
                self.cap.release()
                time.sleep(0.1)
                
                # 재연결 시도
                self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                
                if not self.cap.isOpened():
                    time.sleep(0.5)
                    continue
            
            # 프레임 읽기
            ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.001)
                continue
            
            # 큐에 프레임 추가 (최신 프레임만 유지)
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            # CPU 사용량 조절
            time.sleep(0.0001)
    
    def estimate_distance(self, bbox_height, object_height):
        """객체의 바운딩 박스 높이를 기반으로 거리 추정"""
        if bbox_height == 0:
            return float('inf')
        
        distance = (object_height * self.focal_length) / bbox_height
        return distance
    
    def estimate_age(self, face_roi):
        """얼굴 이미지로부터 나이 추정"""
        # 기본 추정 나이
        estimated_age = 35  # 기본값

        # 얼굴 크기가 너무 작으면 추정하지 않음
        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return estimated_age
        
        try:
            if self.age_model is not None:
                # 사전 처리
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                         (78.4263377603, 87.7689143744, 114.895847746), 
                                         swapRB=False)
                
                # 나이 추정
                self.age_model.setInput(blob)
                age_preds = self.age_model.forward()
                age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
                age_idx = age_preds[0].argmax()
                
                # 연령대의 중간값 사용
                age_range = age_list[age_idx]
                min_age, max_age = map(int, age_range.split('-'))
                estimated_age = (min_age + max_age) // 2
            else:
                # 간단한 추정 방법 (얼굴 크기 기반)
                # 큰 얼굴은 가까이 있는 경향이 있음 = 성인일 가능성
                face_size = face_roi.shape[0] * face_roi.shape[1]
                face_ratio = face_size / (self.width * self.height)
                
                # 매우 간단한 추정 로직 (실제로는 더 정교한 방법 필요)
                if face_ratio > 0.1:  # 큰 얼굴
                    estimated_age = 35  # 성인 추정
                elif face_ratio > 0.05:
                    estimated_age = 25  # 젊은 성인 추정
                elif face_ratio > 0.02:
                    estimated_age = 50  # 중년 추정
                elif face_ratio > 0.01:
                    estimated_age = 15  # 청소년 추정
                else:
                    estimated_age = 10  # 어린이 추정
        
        except Exception as e:
            print(f"나이 추정 오류: {e}")
        
        return estimated_age
    
    def detect_injury(self, person_roi, person_info):
        """사람 영역에서 부상 가능성 탐지"""
        # 기본 부상 가능성 (0~1 사이 값)
        injury_probability = 0.0
        posture = "정상"
        
        try:
            # 1. 사람의 자세 분석
            if self.pose_model is not None:
                # 자세 추정
                pose_results = self.pose_model.predict(person_roi, verbose=False)
                
                if len(pose_results) > 0 and hasattr(pose_results[0], 'keypoints'):
                    keypoints = pose_results[0].keypoints.data
                    
                    if len(keypoints) > 0:
                        # 키포인트 분석 (머리, 어깨, 엉덩이, 무릎, 발목 등)
                        
                        # 예: 사람이 누워있는지 확인 (수평 자세)
                        # 실제로는 더 복잡한 로직이 필요
                        if len(keypoints) >= 17:  # COCO 모델은 17개 키포인트
                            # 예시: 어깨와 엉덩이 키포인트의 y좌표 비교
                            left_shoulder = keypoints[5]
                            right_shoulder = keypoints[6]
                            left_hip = keypoints[11]
                            right_hip = keypoints[12]
                            
                            # 어깨와 엉덩이가 비슷한 높이에 있으면 누워있을 가능성
                            if abs(left_shoulder[1] - left_hip[1]) < 20 or abs(right_shoulder[1] - right_hip[1]) < 20:
                                injury_probability += 0.7
                                posture = "쓰러짐"
            
            # 2. 움직임 분석 (프레임 간 차이)
            if person_info.get('prev_position') is not None:
                prev_x, prev_y = person_info['prev_position']
                curr_x = (person_info['bbox'][0] + person_info['bbox'][2]) / 2
                curr_y = (person_info['bbox'][1] + person_info['bbox'][3]) / 2
                
                # 움직임 크기 계산
                movement = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                # 움직임이 매우 적으면 부상 가능성 증가
                if movement < 5:
                    person_info['not_moving_frames'] = person_info.get('not_moving_frames', 0) + 1
                    
                    # 장시간 움직임 없음
                    if person_info['not_moving_frames'] > 50:  # 약 2초 이상
                        injury_probability += 0.4
                        if posture == "정상":
                            posture = "움직임 없음"
                else:
                    person_info['not_moving_frames'] = 0
            
            # 현재 위치 저장
            person_info['prev_position'] = [(person_info['bbox'][0] + person_info['bbox'][2]) / 2,
                                          (person_info['bbox'][1] + person_info['bbox'][3]) / 2]
            
            # 3. 자세가 갑자기 변했는지 확인 (이전 자세와 비교)
            if posture != person_info.get('prev_posture', "정상") and posture == "쓰러짐":
                injury_probability += 0.2
            
            person_info['prev_posture'] = posture
            
            # 최종 부상 확률 제한 (0~1 사이)
            injury_probability = min(max(injury_probability, 0.0), 1.0)
            
        except Exception as e:
            print(f"부상 탐지 오류: {e}")
        
        return injury_probability, posture
    
    def get_age_group(self, age):
        """나이대를 구분하여 반환"""
        for (min_age, max_age), group_name in self.age_groups.items():
            if min_age <= age <= max_age:
                return group_name
        return "성인"  # 기본값
        
    def send_to_emergency_api(self, emergency_data):
        """소방청 API로 데이터 전송"""
        if self.api_endpoint is None:
            return False, "API 엔드포인트가 설정되지 않았습니다"
        
        try:
            # API 호출 간격 제한 (너무 자주 호출하지 않도록)
            current_time = time.time()
            if current_time - self.last_api_call_time < self.api_call_interval:
                return False, "API 호출 주기 제한"
            
            self.last_api_call_time = current_time
            
            # JSON 데이터 준비
            payload = {
                "timestamp": datetime.now().isoformat(),
                "location": "카메라_위치_정보",  # 실제 카메라 위치 정보로 대체
                "emergency_level": self._calculate_emergency_level(emergency_data),
                "detected_persons": emergency_data
            }
            
            # API 호출
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"  # 실제 API 키로 대체
            }
            
            response = requests.post(self.api_endpoint, 
                                    json=payload, 
                                    headers=headers,
                                    timeout=5)  # 5초 타임아웃
            
            if response.status_code == 200:
                print(f"API 전송 성공: {len(emergency_data)}명 정보 전송")
                return True, "전송 성공"
            else:
                print(f"API 전송 실패: 상태 코드 {response.status_code}")
                return False, f"API 오류: {response.status_code}"
                
        except requests.RequestException as e:
            print(f"API 통신 오류: {e}")
            return False, f"통신 오류: {str(e)}"
        except Exception as e:
            print(f"API 전송 중 예외 발생: {e}")
            return False, f"예외: {str(e)}"
    
    def _calculate_emergency_level(self, person_data):
        """응급 상황 수준 계산 (0-5, 5가 가장 심각)"""
        if not person_data:
            return 0
        
        # 부상 가능성이 높은 사람 수
        injured_count = sum(1 for p in person_data if p['injury_probability'] > 0.6)
        
        # 어린이/노인 비율
        vulnerable_count = sum(1 for p in person_data 
                              if p['age_group'] in ["어린이", "노인"] and p['injury_probability'] > 0.3)
        
        # 위급 수준 계산
        total_persons = len(person_data)
        
        if injured_count > 5 or (vulnerable_count > 2 and injured_count > 0):
            return 5  # 매우 심각
        elif injured_count > 2 or (vulnerable_count > 0 and injured_count > 0):
            return 4  # 심각
        elif injured_count > 0:
            return 3  # 중간
        elif vulnerable_count > 0:
            return 2  # 주의
        elif total_persons > 10:
            return 1  # 약간 주의
        else:
            return 0  # 정상
    
    def log_detection(self, person_id, bbox, estimated_age, injury_probability, posture, sent_to_api):
        """감지 정보를 로그 파일에 기록"""
        timestamp = datetime.now().isoformat()
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp},{person_id},{x},{y},{w},{h},{estimated_age},"
                    f"{injury_probability:.2f},{posture},{sent_to_api}\n")
    
    def process_frames(self):
        """프레임 처리 및 화면 표시 메인 함수"""
        # 윈도우 생성
        cv2.namedWindow('Emergency Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emergency Detection', self.display_width, self.display_height)
        
        # 키 입력 안내 출력
        self.print_help()
        
        # 객체 수 카운트 변수
        detected_objects = {}
        
        # API 전송 타이머
        last_api_send_time = time.time()
        
        while not self.stopped:
            try:
                # 프레임 가져오기
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame = self.frame_queue.get_nowait()
                
                # 프레임 카운터 증가
                self.counter += 1
                process_this_frame = (self.counter % self.process_every_n_frames == 0)
                
                if self.processed_frame is None or process_this_frame:
                    # 현재 시간 표시
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, current_time, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 현재 프레임의 사람 정보 저장 (API 전송용)
                    current_persons_data = []
                    
                    # 추적 중인 사람 목록 업데이트 (타임아웃 처리)
                    for person_id in list(self.tracked_persons.keys()):
                        self.tracked_persons[person_id]['frames_since_detection'] += 1
                        if self.tracked_persons[person_id]['frames_since_detection'] > self.person_tracking_timeout:
                            del self.tracked_persons[person_id]
                    
                    # 객체 감지 수행
                    if self.detection_enabled:
                        # 객체 카운트 초기화
                        detected_objects = {}
                        
                        # YOLOv8로 객체 감지 실행
                        results = self.model.predict(
                            frame, 
                            conf=self.detection_confidence, 
                            verbose=False,
                            stream=False
                        )
                        
                        # 감지 결과 처리
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            
                            # 모든 감지된 객체 처리
                            for box in boxes:
                                # 바운딩 박스 및 정보 추출
                                box_xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
                                x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
                                confidence = box.conf[0].item()
                                class_id = int(box.cls[0].item())
                                class_name = self.classes[class_id]
                                
                                # 사람 객체만 처리
                                if class_name == 'person':
                                    # 객체 높이 및 거리 계산
                                    bbox_height = y2 - y1
                                    object_height = self.default_object_sizes.get(class_name, self.default_object_sizes['default'])
                                    distance = self.estimate_distance(bbox_height, object_height)
                                    
                                    # 사람 이미지 영역 추출
                                    person_roi = frame[max(0, y1):min(y2, self.height), max(0, x1):min(x2, self.width)]
                                    
                                    # 얼굴 감지
                                    estimated_age = 30  # 기본값
                                    if self.face_detector is not None and person_roi.size > 0:
                                        try:
                                            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                                            faces = self.face_detector.detectMultiScale(gray_roi, 1.3, 5)
                                            
                                            if len(faces) > 0:
                                                for (fx, fy, fw, fh) in faces:
                                                    face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                                                    if face_roi.size > 0:
                                                        estimated_age = self.estimate_age(face_roi)
                                        except Exception as e:
                                            print(f"얼굴 감지 오류: {e}")
                                    
                                    # 연령대 결정
                                    age_group = self.get_age_group(estimated_age)
                                    
                                    # 사람 ID 할당 (간단한 추적)
                                    person_id = None
                                    min_distance = float('inf')
                                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                    
                                    # 기존 추적 중인 사람과 매칭
                                    for pid, pinfo in self.tracked_persons.items():
                                        if pinfo['frames_since_detection'] <= self.person_tracking_timeout:
                                            p_center_x = (pinfo['bbox'][0] + pinfo['bbox'][2]) // 2
                                            p_center_y = (pinfo['bbox'][1] + pinfo['bbox'][3]) // 2
                                            
                                            dist = math.sqrt((center_x - p_center_x)**2 + (center_y - p_center_y)**2)
                                            
                                            # 거리가 일정 범위 내에 있으면 같은 사람으로 판단
                                            if dist < min_distance and dist < 100:  # 임계값
                                                min_distance = dist
                                                person_id = pid
                                    
                                    # 새로운 사람이면 ID 생성
                                    if person_id is None:
                                        person_id = self.next_person_id
                                        self.next_person_id += 1
                                        self.tracked_persons[person_id] = {
                                            'bbox': (x1, y1, x2, y2),
                                            'frames_since_detection': 0,
                                            'estimated_age': estimated_age,
                                            'age_group': age_group,
                                            'injury_probability': 0.0,
                                            'posture': "정상",
                                            'detection_count': 1
                                        }
                                    # 기존 추적 중인 사람 정보 업데이트
                                    else:
                                        self.tracked_persons[person_id]['bbox'] = (x1, y1, x2, y2)
                                        self.tracked_persons[person_id]['frames_since_detection'] = 0
                                        self.tracked_persons[person_id]['detection_count'] += 1
                                        
                                        # 나이 추정치 업데이트 (이동 평균)
                                        prev_age = self.tracked_persons[person_id]['estimated_age']
                                        new_count = self.tracked_persons[person_id]['detection_count']
                                        
                                        # 가중치 이동 평균 계산
                                        updated_age = (prev_age * (new_count - 1) + estimated_age) / new_count
                                        self.tracked_persons[person_id]['estimated_age'] = updated_age
                                        self.tracked_persons[person_id]['age_group'] = self.get_age_group(updated_age)
                                    
                                    # 부상 여부 탐지
                                    injury_probability, posture = self.detect_injury(
                                        person_roi, self.tracked_persons[person_id]
                                    )
                                    
                                    # 부상 확률 업데이트
                                    self.tracked_persons[person_id]['injury_probability'] = max(
                                        injury_probability,
                                        self.tracked_persons[person_id].get('injury_probability', 0.0)
                                    )
                                    self.tracked_persons[person_id]['posture'] = posture
                                    
                                    # 객체 수 카운트
                                    if class_name in detected_objects:
                                        detected_objects[class_name] += 1
                                    else:
                                        detected_objects[class_name] = 1
                                    
                                    # 색상 결정 (부상 가능성에 따라)
                                    # 부상 가능성: 높음 = 빨강, 중간 = 주황, 낮음 = 초록
                                    injury_prob = self.tracked_persons[person_id]['injury_probability']
                                    if injury_prob > 0.6:
                                        color = (0, 0, 255)  # 빨강 (RGB)
                                    elif injury_prob > 0.3:
                                        color = (0, 165, 255)  # 주황
                                    else:
                                        color = (0, 255, 0)  # 초록
                                    
                                    # 바운딩 박스 그리기
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # ID, 나이, 부상 확률 표시
                                    text = f"ID:{person_id} {age_group}({int(self.tracked_persons[person_id]['estimated_age'])})"
                                    
                                    # 텍스트 배경 영역
                                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                                    
                                    # 텍스트
                                    cv2.putText(frame, text, (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    
                                    # 두 번째 텍스트 줄 (부상 확률)
                                    injury_text = f"상태:{posture} ({injury_prob:.2f})"
                                    (inj_width, inj_height), _ = cv2.getTextSize(injury_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(frame, (x1, y1 - text_height - inj_height - 12), 
                                                (x1 + inj_width, y1 - text_height - 8), color, -1)
                                    cv2.putText(frame, injury_text, (x1, y1 - text_height - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    
                                    # 현재 사람 정보 저장 (API 전송용)
                                    person_data = {
                                        "person_id": person_id,
                                        "bbox": [x1, y1, x2, y2],
                                        "estimated_age": int(self.tracked_persons[person_id]['estimated_age']),
                                        "age_group": age_group,
                                        "injury_probability": injury_prob,
                                        "posture": posture,
                                        "distance": distance
                                    }
                                    current_persons_data.append(person_data)
                                    
                                    # 로그 파일에 기록
                                    self.log_detection(
                                        person_id, (x1, y1, x2, y2), 
                                        self.tracked_persons[person_id]['estimated_age'],
                                        injury_prob, posture, 
                                        False  # API 전송 여부는 나중에 업데이트
                                    )
                                else:
                                    # 다른 객체 (사람이 아닌 경우)
                                    if class_name in detected_objects:
                                        detected_objects[class_name] += 1
                                    else:
                                        detected_objects[class_name] = 1
                                    
                                    # 색상 가져오기
                                    color = self.color_map.get(class_name, (0, 255, 0))
                                    
                                    # 바운딩 박스 그리기
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # 객체 이름 표시
                                    text = f"{class_name}: {confidence:.2f}"
                                    
                                    # 텍스트 배경 영역
                                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                                    
                                    # 텍스트
                                    cv2.putText(frame, text, (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                    
                    # API 전송 (소방청 데이터 전송)
                    current_time = time.time()
                    if self.api_endpoint and current_time - last_api_send_time >= self.api_call_interval:
                        if current_persons_data:
                            # 부상 가능성이 있는 사람이 있는 경우에만 API 전송
                            injured_persons = [p for p in current_persons_data if p['injury_probability'] > 0.3]
                            
                            if injured_persons:
                                success, message = self.send_to_emergency_api(current_persons_data)
                                
                                if success:
                                    # 성공 메시지 표시
                                    cv2.putText(frame, f"소방청 데이터 전송 완료: {len(current_persons_data)}명", 
                                            (self.width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.6, (0, 255, 0), 2)
                                    
                                    # 로그 파일 업데이트 (API 전송 여부)
                                    for person in current_persons_data:
                                        # TODO: 로그 업데이트 로직
                                        pass
                                else:
                                    # 실패 메시지 표시
                                    cv2.putText(frame, f"소방청 데이터 전송 실패: {message}", 
                                            (self.width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.6, (0, 0, 255), 2)
                        
                        last_api_send_time = current_time
                    
                    # 감지 상태 표시
                    if self.detection_enabled:
                        cv2.putText(frame, "Detection: ON", (self.width - 180, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Detection: OFF", (self.width - 180, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 객체 수 표시
                    person_count = detected_objects.get('person', 0)
                    cv2.putText(frame, f"감지된 사람: {person_count}명", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 부상 가능성 정보 표시
                    injured_count = 0
                    for person_id, info in self.tracked_persons.items():
                        if info['injury_probability'] > 0.6 and info['frames_since_detection'] < 10:
                            injured_count += 1
                    
                    if injured_count > 0:
                        injury_text = f"위급 상황 감지: {injured_count}명"
                        cv2.putText(frame, injury_text, (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 녹화 중인 경우 표시
                    if self.recording:
                        cv2.putText(frame, "REC", (self.width - 70, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    self.processed_frame = frame
                
                # 녹화 중인 경우 저장
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(self.processed_frame)
                
                # 화면 표시
                cv2.imshow('Emergency Detection', self.processed_frame)
                
                # 키 입력 처리 (최소 대기 시간)
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_events(key):
                    break
                
            except Exception as e:
                print(f"프레임 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        # 자원 해제
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
    
    def print_help(self):
        """키 입력 안내 출력"""
        print("\n============ 응급 상황 감지 시스템 ============")
        print("q: 프로그램 종료")
        print("s: 현재 화면 스냅샷 저장")
        print("r: 녹화 시작/중지")
        print("d: 객체 감지 활성화/비활성화")
        print("+/-: 감지 신뢰도 조정 (0.05)")
        print("f/g: 프레임 처리 주기 조정")
        print("e: 소방청 데이터 수동 전송")
        print("l: 로그 파일 위치 출력")
        print("h: 도움말 표시")
        print("===================================================\n")
    
    def handle_key_events(self, key):
        """키 입력 처리"""
        # q: 종료
        if key == ord('q'):
            return False
            
        # s: 스냅샷 저장
        elif key == ord('s'):
            self.save_snapshot()
            
        # r: 녹화 시작/종료
        elif key == ord('r'):
            self.toggle_recording()
        
        # d: 객체 감지 활성화/비활성화
        elif key == ord('d'):
            self.detection_enabled = not self.detection_enabled
            print(f"객체 감지: {'활성화' if self.detection_enabled else '비활성화'}")
        
        # +/-: 감지 신뢰도 조정
        elif key == ord('+') or key == ord('='):
            self.detection_confidence = min(self.detection_confidence + 0.05, 0.95)
            print(f"감지 신뢰도 임계값: {self.detection_confidence:.2f}")
        elif key == ord('-'):
            self.detection_confidence = max(self.detection_confidence - 0.05, 0.05)
            print(f"감지 신뢰도 임계값: {self.detection_confidence:.2f}")
            
        # f/g: 프레임 처리 주기 조정
        elif key == ord('f'):
            self.process_every_n_frames = min(self.process_every_n_frames + 1, 10)
            print(f"매 {self.process_every_n_frames}번째 프레임만 처리")
        elif key == ord('g'):
            self.process_every_n_frames = max(self.process_every_n_frames - 1, 1)
            print(f"매 {self.process_every_n_frames}번째 프레임만 처리")
        
        # e: 소방청 데이터 수동 전송
        elif key == ord('e'):
            if self.api_endpoint:
                persons_data = [
                    {
                        "person_id": pid,
                        "bbox": list(info['bbox']),
                        "estimated_age": int(info['estimated_age']),
                        "age_group": info['age_group'],
                        "injury_probability": info['injury_probability'],
                        "posture": info['posture'],
                        "distance": 0  # 간단한 예시
                    }
                    for pid, info in self.tracked_persons.items()
                    if info['frames_since_detection'] < 10  # 최근에 감지된 사람만
                ]
                
                if persons_data:
                    success, message = self.send_to_emergency_api(persons_data)
                    print(f"수동 데이터 전송 결과: {'성공' if success else '실패'} - {message}")
                else:
                    print("전송할 데이터가 없습니다.")
            else:
                print("API 엔드포인트가 설정되지 않았습니다.")
        
        # l: 로그 파일 위치 출력
        elif key == ord('l'):
            print(f"로그 파일 위치: {os.path.abspath(self.log_file)}")
        
        # h: 도움말 표시
        elif key == ord('h'):
            self.print_help()
        
        return True
    
    def save_snapshot(self):
        """현재 화면 스냅샷 저장"""
        if self.processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"{self.save_dir}/snapshot_{timestamp}.jpg"
            cv2.imwrite(img_path, self.processed_frame)
            print(f"스냅샷 저장됨: {img_path}")
    
    def toggle_recording(self):
        """녹화 시작/종료"""
        if not self.recording:
            # 녹화 시작
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"{self.save_dir}/video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, 
                                              (self.width, self.height))
            self.recording = True
            print(f"녹화 시작: {video_path}")
        else:
            # 녹화 종료
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("녹화 종료")
    
    def stop(self):
        """리소스 정리 및 종료"""
        # 플래그 설정
        self.stopped = True
        
        # 스레드 종료 대기
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # 캡처 객체 해제
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # 녹화 종료
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        print("프로그램이 종료되었습니다.")

def check_model_files():
    """필요한 모델 파일이 있는지 확인하고 안내"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # age_net.caffemodel 확인
    age_model_path = os.path.join(models_dir, "age_net.caffemodel")
    if not os.path.exists(age_model_path):
        print("\n" + "="*60)
        print("경고: age_net.caffemodel 파일이 필요합니다.")
        print("다음 URL에서 다운로드 가능합니다:")
        print("1. https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel")
        print("2. https://drive.google.com/file/d/1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW/view?usp=sharing")
        print(f"다운로드 후 '{models_dir}' 폴더에 저장하세요.")
        print("="*60 + "\n")
        
        # 사용자가 파일을 다운로드할 시간을 줌
        user_input = input("계속하려면 Enter 키를 누르세요. 취소하려면 'q'를 입력하세요: ")
        if user_input.lower() == 'q':
            return False
    
    # YOLOv8 pose 모델 확인
    try:
        from ultralytics import YOLO
        # 모델 존재 확인 (다운로드됨)
        YOLO("yolov8n-pose.pt")
    except Exception as e:
        print(f"\n경고: YOLOv8 pose 모델 로드 중 오류: {e}")
        print("자세 추정 기능이 제한될 수 있습니다.")
    
    return True

def main():
    # 필요한 모델 파일 확인
    if not check_model_files():
        print("필요한 모델 파일이 없어 프로그램을 종료합니다.")
        return
    
    # RTMP 주소 설정
    rtmp_url = "rtmp://11.16.45.11/live/test"
    
    # 소방청 API 엔드포인트 (예시)
    api_endpoint = None  # 실제 소방청 API 주소로 변경 필요
    
    try:
        # 응급 감지 객체 생성
        detector = EmergencyDetector(rtmp_url, api_endpoint)
        
        # RTMP 스트림 연결 시작
        if not detector.start_capture():
            print("스트림 연결에 실패했습니다.")
            return
        
        # 모델 로드
        if not detector.load_models():
            print("모델 로드에 실패했습니다.")
            detector.stop()
            return
        
        # 키보드 단축키 안내
        detector.print_help()
        
        # 프레임 처리 시작
        detector.process_frames()
    
    except KeyboardInterrupt:
        print("사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # 자원 해제
        if 'detector' in locals():
            detector.stop()

if __name__ == "__main__":
    main()