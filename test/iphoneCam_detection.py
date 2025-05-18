import cv2
import torch
import time
import sys
import os
import numpy as np
import argparse

def find_iphone_camera():
    """
    아이폰 카메라를 자동으로 찾습니다.
    맥북 내장 카메라를 제외하고 연결된 카메라를 찾습니다.
    """
    print("[INFO] 아이폰 카메라 찾는 중...")
    
    # 우선 모든 카메라 탐색
    max_cameras_to_check = 10
    available_cameras = []
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_name = f"카메라 {i}"
                try:
                    camera_name = cap.getBackendName()
                except:
                    pass
                
                # 카메라 정보 저장
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append((i, camera_name, width, height))
            cap.release()
    
    if not available_cameras:
        print("[오류] 사용 가능한 카메라를 찾을 수 없습니다.")
        return None
    
    # 여러 카메라가 있는 경우, 맥북 내장 카메라(일반적으로 인덱스 0)를 제외한 
    # 첫 번째 카메라를 아이폰 카메라로 간주
    if len(available_cameras) > 1:
        # 맥북 내장 카메라가 아닌 첫 번째 카메라 선택 (일반적으로 인덱스 1 이상)
        for camera in available_cameras:
            if camera[0] > 0:  # 인덱스 0이 아닌 카메라
                print(f"[INFO] 아이폰 카메라 발견: 인덱스 {camera[0]}, 해상도: {camera[2]}x{camera[3]}")
                return camera[0]
    
    # 만약 카메라가 하나만 있거나 다른 카메라를 찾지 못한 경우, 사용 가능한 첫 번째 카메라 사용
    print(f"[INFO] 단일 카메라 발견: 인덱스 {available_cameras[0][0]}, 해상도: {available_cameras[0][2]}x{available_cameras[0][3]}")
    return available_cameras[0][0]

def detect_iphone_camera(model_path='yolov5s.pt', conf_threshold=0.5, reconnect_attempts=5):
    """
    아이폰 카메라를 사용하여 실시간 객체 탐지를 수행합니다.
    
    Args:
        model_path: 사용할 YOLOv5 모델 파일 경로 (기본값: 'yolov5s.pt')
        conf_threshold: 객체 탐지 신뢰도 임계값 (기본값: 0.5)
        reconnect_attempts: 카메라 연결 끊김 시 재시도 횟수 (기본값: 5)
    """
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 장치: {device}")
    
    # YOLOv5 소스코드 경로 추가 (로컬의 yolov5 폴더 사용)
    yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
        sys.path.append(yolov5_path)
        print(f"[INFO] YOLOv5 경로 추가됨: {yolov5_path}")
    
    # 모델 로드
    print(f"[INFO] 모델 '{model_path}' 로드 중...")
    
    try:
        # YOLOv5 모듈 임포트
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.general import non_max_suppression
        
        # 모델 로드
        model = attempt_load(model_path, device=device)
        model.eval()
        
        # 모델이 half 정밀도인지 확인
        is_half = next(model.parameters()).dtype == torch.float16
        if is_half:
            print("[INFO] 모델이 FP16(half precision) 형식입니다.")
        else:
            print("[INFO] 모델이 FP32(float precision) 형식입니다.")
        
        class_names = model.names
            
    except Exception as e:
        print(f"[오류] 모델 로드 중 예외 발생: {str(e)}")
        return
    
    # 아이폰 카메라 찾기 (자동으로 찾기)
    camera_index = find_iphone_camera()
    if camera_index is None:
        print("[오류] 아이폰 카메라를 찾을 수 없습니다.")
        return
    
    print(f"[INFO] 아이폰 카메라 (인덱스: {camera_index}) 초기화 중...")
    
    # 이미지 크기 설정
    img_size = 640
    
    print("[INFO] 실시간 객체 탐지 시작. 종료하려면 'q'키를 누르세요.")
    
    # 성능 측정을 위한 변수
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # 카메라 연결 및 초기화
    cap = None
    connected = False
    reconnect_count = 0
    last_reconnect_time = 0
    connection_stable = False
    
    try:
        while True:
            # 카메라 연결 상태 확인 및 재연결
            current_time = time.time()
            if not connected or (cap is not None and not cap.isOpened()):
                # 카메라가 연결되지 않았거나 연결이 끊어진 경우
                
                # 이전 연결 종료
                if cap is not None:
                    cap.release()
                
                # 재연결 시도 간격 조정 (너무 빠른 재연결 방지)
                if current_time - last_reconnect_time < 2.0:  # 2초 간격으로 재연결 시도
                    time.sleep(0.5)
                    continue
                
                # 재연결 시도 횟수 제한
                if reconnect_count >= reconnect_attempts:
                    print("\n[오류] 아이폰 카메라 연결을 안정적으로 유지할 수 없습니다.")
                    print("다음 사항을 확인해 보세요:")
                    print("1. 아이폰과 맥북이 동일한 Apple ID로 로그인되어 있는지 확인")
                    print("2. 두 기기가 충분히 가까이 있는지 확인 (1m 이내)")
                    print("3. 블루투스와 Wi-Fi가 두 기기 모두에서 활성화되어 있는지 확인")
                    print("4. 설정 > 일반 > AirDrop 및 핸드오프에서 핸드오프 기능이 활성화되어 있는지 확인")
                    print("5. 카메라 연결에 문제가 있으면 서드파티 앱(Camo, EpocCam 등)을 사용해 보세요")
                    return
                
                print(f"\n[INFO] 아이폰 카메라 연결 시도 중... (시도 {reconnect_count+1}/{reconnect_attempts})")
                
                # 아이폰 카메라를 다시 찾고 연결 시도
                camera_index = find_iphone_camera()
                if camera_index is None:
                    print("[경고] 아이폰 카메라를 찾을 수 없습니다. 재시도 중...")
                    reconnect_count += 1
                    last_reconnect_time = current_time
                    time.sleep(2)
                    continue
                
                # 카메라에 연결 시도
                cap = cv2.VideoCapture(camera_index)
                last_reconnect_time = current_time
                reconnect_count += 1
                
                if cap.isOpened():
                    # 카메라 해상도 설정 (처음에는 낮은 해상도로 시작)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # 실제 설정된 해상도 확인
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[INFO] 아이폰 카메라 연결됨. 해상도: {actual_width}x{actual_height}")
                    
                    connected = True
                    # 연결 성공 시 초기 프레임 10개 버리기 (초기 프레임이 불안정할 수 있음)
                    for _ in range(10):
                        cap.read()
                else:
                    print("[경고] 아이폰 카메라 연결 실패. 재시도 중...")
                    time.sleep(1)
                    continue
            
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("[경고] 프레임을 가져올 수 없습니다. 카메라 연결 확인 중...")
                connected = False
                continue
            
            # 안정적 연결 표시 - 첫 30개 프레임을 성공적으로 가져오면 연결이 안정적이라고 판단
            if not connection_stable and frame_count > 30:
                connection_stable = True
                # 연결이 안정적이면 해상도를 높일 수 있음 (선택 사항)
                if actual_width < 1920:
                    print("[INFO] 연결이 안정적입니다. 해상도를 높이는 중...")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[INFO] 해상도 업데이트됨: {actual_width}x{actual_height}")
                
                print("[INFO] 아이폰 카메라 연결이 안정적입니다.")
            
            # 원본 이미지 복사
            original_img = frame.copy()
            
            # 이미지 전처리
            img = cv2.resize(frame, (img_size, img_size))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            
            # 데이터 타입 설정 (FP16 또는 FP32)
            if is_half:
                img = img.half()
            else:
                img = img.float()
                
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # 배치 차원 추가
            
            # 객체 탐지
            with torch.no_grad():
                pred = model(img)[0]
                # NMS 적용
                pred = non_max_suppression(pred, conf_thres=conf_threshold)
                pred = pred[0] if len(pred) > 0 else torch.zeros((0, 6), device=device)
            
            # 결과 시각화
            if len(pred):
                # 좌표 변환 (모델 입력 크기 -> 원본 이미지 크기)
                pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], original_img.shape).round()
                
                # 각 탐지 결과 처리
                for *xyxy, conf, cls_id in pred:
                    if conf >= conf_threshold:
                        # 좌표 추출
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # 클래스 이름 가져오기
                        cls_id = int(cls_id)
                        class_name = class_names[cls_id]
                        
                        # 바운딩 박스 그리기
                        color = (0, 255, 0)  # 녹색
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
                        
                        # 레이블 그리기
                        label = f"{class_name}: {conf:.2f}"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(original_img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                        cv2.putText(original_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # FPS 계산 및 표시
            frame_count += 1
            if frame_count >= 10:  # 10프레임마다 FPS 업데이트
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # 연결 상태 표시
            status_text = "연결 상태: 안정적" if connection_stable else f"연결 상태: 설정 중 ({reconnect_count}/{reconnect_attempts})"
            cv2.putText(original_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(original_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 화면에 표시
            cv2.imshow("iPhone Camera Object Detection", original_img)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 중단되었습니다.")
    
    finally:
        # 자원 해제
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 종료되었습니다.")

# scale_coords 함수 정의
def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # 좌표가 이미지 경계를 넘지 않도록 조정
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    
    return coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="아이폰 카메라를 이용한 실시간 객체 탐지")
    parser.add_argument("--model", type=str, default="yolov5s.pt",
                        help="사용할 YOLOv5 모델 파일 경로 (기본값: 'yolov5s.pt')")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="객체 탐지 신뢰도 임계값 (기본값: 0.5)")
    parser.add_argument("--reconnect", type=int, default=5,
                        help="카메라 연결 끊김 시 재시도 횟수 (기본값: 5)")
    
    args = parser.parse_args()
    
    detect_iphone_camera(args.model, args.conf, args.reconnect)