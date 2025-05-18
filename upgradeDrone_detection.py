import cv2
import time
import os
import numpy as np
from datetime import datetime
from threading import Thread
import queue
import math
from ultralytics import YOLO  # YOLOv8 사용

class SimpleObjectDetector:
    def __init__(self, rtmp_url):
        # 기본 설정
        self.rtmp_url = rtmp_url
        self.frame_queue = queue.Queue(maxsize=1)  # 프레임 큐 크기를 1로 줄임
        self.stopped = False
        
        # 저장 관련 설정
        self.save_dir = "object_detections"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 녹화 관련 설정
        self.recording = False
        self.video_writer = None
        
        # 감지 관련 설정
        self.detection_enabled = True
        self.detection_confidence = 0.35  # 더 낮은 기본값으로 더 많은 객체 감지
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
            'car': 1.5,     # 평균 자동차 높이 (미터)
            'truck': 2.5,   # 평균 트럭 높이 (미터)
            'motorcycle': 1.2,  # 평균 오토바이 높이 (미터)
            'bicycle': 1.0,  # 평균 자전거 높이 (미터)
            'drone': 0.3,    # 평균 드론 크기 (미터)
            'default': 0.5   # 기본 크기
        }
        self.focal_length = None  # 초점 거리 (픽셀) - 캘리브레이션 시 계산
        self.camera_fov = 75.0   # 카메라 시야각 (도)
        
        # 색상 맵 (클래스별 색상)
        self.color_map = {}

    def start_capture(self):
        """RTMP 스트림 캡처 시작"""
        print(f"RTMP 스트림에 연결 시도 중: {self.rtmp_url}")
        
        # FFMPEG 설정 - 극단적인 저지연 설정
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;0|max_delay;0|fflags;discardcorrupt+nobuffer+fastseek|flags;low_delay"
        
        # 캡처 객체 생성
        self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 버퍼 크기를 0으로 설정
        
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
    
    def load_model(self):
        """YOLOv8 모델 로드"""
        print("YOLOv8 모델 로드 중...")
        
        try:
            # YOLOv8 모델 로드 (Ultralytics 라이브러리 사용)
            self.model = YOLO("yolov8n.pt")  # 더 높은 정확도를 위해 yolov8x.pt 사용 가능
            
            # 모델 클래스 목록 가져오기
            self.classes = self.model.names
            
            # 클래스별 랜덤 색상 생성
            np.random.seed(42)  # 일관된 색상을 위한 시드 설정
            colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
            
            # 색상 맵 생성
            for i, class_name in self.classes.items():
                self.color_map[class_name] = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            
            print(f"YOLOv8 모델 로드 완료 (클래스 수: {len(self.classes)})")
            return True
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
    
    def _capture_frames(self):
        """프레임 캡처 스레드 함수 (저지연 최적화)"""
        while not self.stopped:
            if not self.cap.isOpened():
                print("캡처 연결이 끊어졌습니다. 재연결 시도...")
                self.cap.release()
                time.sleep(0.1)  # 재연결 대기 시간 감소
                
                # 재연결 시도
                self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                
                if not self.cap.isOpened():
                    time.sleep(0.5)  # 재연결 실패 시 대기 시간 감소
                    continue
            
            # 프레임 읽기
            ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.001)  # 프레임 읽기 실패 시 대기 시간 최소화
                continue
            
            # 큐에 프레임 추가 (최신 프레임만 유지)
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # 오래된 프레임 제거
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            # 스레드 CPU 사용량 조절 (최소화)
            time.sleep(0.0001)
    
    def estimate_distance(self, bbox_height, object_height):
        """객체의 바운딩 박스 높이를 기반으로 거리 추정"""
        # 거리 = (알려진 실제 높이 * 초점 거리) / 픽셀 높이
        if bbox_height == 0:
            return float('inf')
        
        distance = (object_height * self.focal_length) / bbox_height
        return distance
    
    def process_frames(self):
        """프레임 처리 및 화면 표시 메인 함수"""
        # 윈도우 생성
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', self.display_width, self.display_height)
        
        # 키 입력 안내 출력
        self.print_help()
        
        # 객체 수 카운트 변수
        detected_objects = {}
        
        while not self.stopped:
            try:
                # 프레임 가져오기
                if self.frame_queue.empty():
                    time.sleep(0.001)  # 대기 시간 최소화
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
                    
                    # 객체 감지 수행
                    if self.detection_enabled:
                        # 객체 카운트 초기화
                        detected_objects = {}
                        
                        # YOLOv8로 객체 감지 실행 (수정된 부분)
                        results = self.model.predict(
                            frame, 
                            conf=self.detection_confidence, 
                            verbose=False,
                            stream=False  # stream 옵션 비활성화
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
                                
                                # 객체 높이 및 거리 계산
                                bbox_height = y2 - y1
                                object_height = self.default_object_sizes.get(class_name, self.default_object_sizes['default'])
                                distance = self.estimate_distance(bbox_height, object_height)
                                
                                # 객체 수 카운트
                                if class_name in detected_objects:
                                    detected_objects[class_name] += 1
                                else:
                                    detected_objects[class_name] = 1
                                
                                # 색상 가져오기
                                color = self.color_map.get(class_name, (0, 255, 0))
                                
                                # 바운딩 박스 그리기
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # 레이블 및 거리 정보 표시 (투명 배경)
                                text = f"{class_name}: {confidence:.2f}, {distance:.1f}m"
                                
                                # 텍스트 배경 영역
                                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                                
                                # 텍스트
                                cv2.putText(frame, text, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # 감지 상태 표시
                    if self.detection_enabled:
                        cv2.putText(frame, "Detection: ON", (self.width - 180, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Detection: OFF", (self.width - 180, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 객체 수 표시
                    cv2.putText(frame, f"Objects Detected: {sum(detected_objects.values())}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 감지된 객체 목록 표시
                    y_offset = 90
                    for class_name, count in detected_objects.items():
                        color = self.color_map.get(class_name, (0, 255, 0))
                        cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
                    
                    # 녹화 중인 경우 표시
                    if self.recording:
                        cv2.putText(frame, "REC", (self.width - 70, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    self.processed_frame = frame
                
                # 녹화 중인 경우 저장
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(self.processed_frame)
                
                # 화면 표시
                cv2.imshow('Object Detection', self.processed_frame)
                
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
        print("\n============ 객체 감지 및 거리 측정 시스템 (초저지연) ============")
        print("q: 프로그램 종료")
        print("s: 현재 화면 스냅샷 저장")
        print("r: 녹화 시작/중지")
        print("d: 객체 감지 활성화/비활성화")
        print("+/-: 감지 신뢰도 조정 (0.05)")
        print("f/g: 프레임 처리 주기 조정")
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

def main():
    # RTMP 주소 설정
    rtmp_url = "rtmp://11.16.45.11/live/test"
    
    try:
        # 객체 감지 객체 생성
        detector = SimpleObjectDetector(rtmp_url)
        
        # RTMP 스트림 연결 시작
        if not detector.start_capture():
            print("스트림 연결에 실패했습니다.")
            return
        
        # YOLOv8 모델 로드
        if not detector.load_model():
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