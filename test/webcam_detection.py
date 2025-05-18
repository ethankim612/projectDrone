import cv2
import torch
import time
import sys
import os

def detect_webcam(model_path='yolov5s.pt', conf_threshold=0.5):
    """
    맥북 카메라를 사용하여 실시간 객체 탐지를 수행합니다.
    
    Args:
        model_path: 사용할 YOLOv5 모델 파일 경로 (기본값: 'yolov5s.pt')
        conf_threshold: 객체 탐지 신뢰도 임계값 (기본값: 0.5)
    """
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 장치: {device}")
    
    # YOLOv5 소스코드 경로 추가
    yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
        sys.path.append(yolov5_path)
        print(f"[INFO] YOLOv5 경로 추가됨: {yolov5_path}")
    
    # 모델 로드
    print(f"[INFO] 모델 '{model_path}' 로드 중...")
    
    try:
        # YOLOv5 모듈 임포트 시도
        try:
            from models.experimental import attempt_load
            from utils.general import non_max_suppression
            
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
            
        except ImportError:
            # YOLOv5 패키지 사용 시도
            import yolov5
            
            model = yolov5.load(model_path)
            model.conf = conf_threshold
            is_half = False
            class_names = model.names
            
    except Exception as e:
        print(f"[오류] 모델 로드 중 예외 발생: {str(e)}")
        return
    
    # 웹캠 설정
    print("[INFO] 카메라 초기화 중...")
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라 (맥북 내장 카메라)
    
    if not cap.isOpened():
        print("[오류] 카메라를 열 수 없습니다.")
        return
    
    # 카메라 해상도 설정 (선택 사항)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 이미지 크기 설정
    img_size = 640
    
    print("[INFO] 실시간 객체 탐지 시작. 종료하려면 'q'키를 누르세요.")
    
    # 성능 측정을 위한 변수
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 가져올 수 없습니다.")
            break
        
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
            if hasattr(model, 'predict'):  # YOLOv5 패키지인 경우
                results = model(img)
                pred = results.pred[0]
            else:  # 직접 모델 로드한 경우
                pred = model(img)[0]
                # NMS 적용
                pred = non_max_suppression(pred, conf_thres=conf_threshold)
                pred = pred[0]  # 첫 번째 이미지의 결과
        
        # 결과 시각화
        if len(pred):
            # YOLOv5 패키지인 경우 별도 처리 필요 없음
            if not hasattr(model, 'predict'):
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
        
        cv2.putText(original_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 화면에 표시
        cv2.imshow("Object Detection", original_img)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
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
    import numpy as np
    import argparse
    
    parser = argparse.ArgumentParser(description="맥북 카메라를 이용한 실시간 객체 탐지")
    parser.add_argument("--model", type=str, default="yolov5s.pt",
                        help="사용할 YOLOv5 모델 파일 경로 (기본값: 'yolov5s.pt')")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="객체 탐지 신뢰도 임계값 (기본값: 0.5)")
    
    args = parser.parse_args()
    
    detect_webcam(args.model, args.conf)