import cv2
import torch
import numpy as np
import time
import os
import sys
import argparse
from pathlib import Path

def analyze_video(video_path, model_path='yolov5s.pt', output_dir='detected_videos', conf_threshold=0.5):
    
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    
    output_path = os.path.join(output_dir, f"{video_name_without_ext}_detected.mp4")

    if not os.path.exists(model_path):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 장치: {device}")
    
    yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
        sys.path.append(yolov5_path)
        print(f"[INFO] YOLOv5 경로 추가됨: {yolov5_path}")
    

    print(f"[INFO] 로컬 모델 '{model_path}' 로드 중...")
    
    try:
        
            from models.experimental import attempt_load
            from utils.general import non_max_suppression
            
           
            model = attempt_load(model_path, device=device)
            
            model.eval()  # 평가 모드로 설정
            print(f"[INFO] YOLOv5 모듈을 통해 모델을 성공적으로 로드했습니다.")
            
        
            if hasattr(model, 'stride'):
                stride = int(model.stride.max())
                print(f"[INFO] 모델 stride: {stride}")
            else:
                stride = 32
                print(f"[INFO] 모델 stride를 기본값({stride})으로 설정")
            
       
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                print("[INFO] 모델이 FP16(half precision) 형식입니다.")
            else:
                print("[INFO] 모델이 FP32(float precision) 형식입니다.")
            
            class_names = model.names
            print(f"[INFO] 모델에서 클래스 이름 로드: {len(class_names)}개")
            print(f"클래스 목록: {class_names}")
            
        
            img_size = 640
            
        except ImportError as e:
            print(f"[경고] YOLOv5 모듈 임포트 오류: {e}")
            print("[INFO] 기본 PyTorch 로더로 대체합니다...")
            
            # PyTorch 기본 로더 사용
            model = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    print("[경고] 모델 가중치만 있는 파일입니다. 완전한 모델이 필요합니다.")
            
            model.to(device).eval()
            
  
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                print("[INFO] 모델이 FP16(half precision) 형식입니다.")
            else:
                print("[INFO] 모델이 FP32(float precision) 형식입니다.")
            
            # 기본 COCO 클래스 레이블
            class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
                56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
            }
            print("[INFO] 기본 COCO 클래스 이름 사용")
            
        
            img_size = 640
            stride = 32
        
    except Exception as e:
        print("   git clone https://github.com/ultralytics/yolov5")
        return None
    

    print(f"[INFO] 비디오 '{video_path}' 처리 중...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[오류] 비디오 파일을 열 수 없습니다: {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    detection_by_class = {}
    start_time = time.time()
    
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    
    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        
        if coords.shape[1] >= 4:  # 좌표 형식이 맞는지 확인
            coords[:, 0].clamp_(0, img0_shape[1])  # x1
            coords[:, 1].clamp_(0, img0_shape[0])  # y1
            coords[:, 2].clamp_(0, img0_shape[1])  # x2
            coords[:, 3].clamp_(0, img0_shape[0])  # y2
        
        return coords
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 탐지 수행
        try:
            
            original_img = frame.copy()
            
            img, ratio, pad = letterbox(frame, new_shape=img_size, stride=stride)
            
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            
            if is_half:
                img = img.half()
            else:
                img = img.float()
                
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)  
            
            with torch.no_grad():
               
                pred = model(img)
                
                if isinstance(pred, list) or isinstance(pred, tuple):
                    pred = pred[0]  
                
                try:
                    
                    pred = non_max_suppression(pred, conf_thres=conf_threshold)
                except:
                    # NMS 직접 구현 (간단한 버전)
                    conf_mask = pred[..., 4] >= conf_threshold
                    pred = pred[conf_mask]
                    pred = [pred]  # 배치 형식으로 변환
            
           
            processed_frame = original_img.copy()
            
            for i, det in enumerate(pred):  # 배치의 각 이미지에 대해
                if len(det):
                    
                    try:
                        
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()
                    except Exception as e:
                        print(f"[경고] 좌표 스케일링 오류: {e}")
                        # 좌표 변환 없이 원본 좌표 사용
                        det[:, :4] = det[:, :4].round()
                    
                
                    for j in range(len(det)):
                        # det[j]의 형식이 [x1, y1, x2, y2, conf, cls] 인지 확인
                        if len(det[j]) >= 6:
                            # 바운딩 박스 좌표와 정보
                            x1, y1, x2, y2 = int(det[j][0]), int(det[j][1]), int(det[j][2]), int(det[j][3])
                            conf = float(det[j][4])
                            cls_id = int(det[j][5])
                            
                            # 좌표가 유효한지 확인
                            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                                # 클래스 이름
                                if isinstance(class_names, dict):
                                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                                else:
                                    class_name = class_names[cls_id]
                                
                                # 통계 업데이트
                                detection_count += 1
                                detection_by_class[class_name] = detection_by_class.get(class_name, 0) + 1
                                
                                # 바운딩 박스 그리기
                                color = (0, 255, 0)  # 녹색
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # 레이블 그리기
                                label = f"{class_name}: {conf:.2f}"
                                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                cv2.rectangle(processed_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                                cv2.putText(processed_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        except Exception as e:
            print(f"[경고] 프레임 처리 중 오류 발생: {str(e)}")
            processed_frame = frame.copy()
        
        # FPS 표시
        current_time = time.time() - start_time
        current_fps = frame_count / current_time if current_time > 0 else 0
        cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 영상 저장
        out.write(processed_frame)
        
        # 진행 상황 표시
        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"[진행] {progress:.1f}% ({frame_count}/{total_frames}) - 현재까지 {detection_count}개 객체 탐지됨")
    
    # 자원 해제
    cap.release()
    out.release()
    
    # 처리 시간
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    # 결과 출력
    print("\n========== 분석 결과 ==========")
    print(f"입력 영상: {video_path}")
    print(f"출력 영상: {output_path}")
    print(f"처리된 프레임: {frame_count}")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 FPS: {avg_fps:.2f}")
    print(f"탐지된 객체 총 수: {detection_count}")
    
    # 클래스별 탐지 결과
    print("\n클래스별 탐지 결과:")
    for class_name, count in sorted(detection_by_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {class_name}: {count}개")
    
    print("\n영상 분석이 완료되었습니다.")
    return output_path

def process_videos_in_folder(input_folder, model_path='yolov5s.pt', output_dir='detected_videos', conf_threshold=0.5):
    """
    지정된 폴더 내의 모든 영상 파일을 처리합니다.
    
    Args:
        input_folder: 입력 영상 파일이 있는 폴더 경로
        model_path: 사용할 YOLOv5 모델 파일 경로
        output_dir: 결과 영상 저장 디렉토리
        conf_threshold: 객체 탐지 신뢰도 임계값
    """
    # 지원되는 비디오 파일 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # 폴더 내 비디오 파일 찾기
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(input_folder).glob(f'*{ext}')))
    
    if not video_files:
        print(f"[경고] '{input_folder}' 폴더에 비디오 파일이 없습니다.")
        return
    
    print(f"[INFO] '{input_folder}' 폴더에서 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # 각 비디오 파일 처리
    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] 비디오 파일 처리 중: {video_path}")
        analyze_video(str(video_path), model_path, output_dir, conf_threshold)

def process_youtube_video(youtube_url, model_path='yolov5s.pt', output_dir='detected_videos', conf_threshold=0.5):
    """
    유튜브 영상을 다운로드하고 객체 탐지를 수행합니다.
    
    Args:
        youtube_url: 유튜브 영상 URL
        model_path: 사용할 YOLOv5 모델 파일 경로
        output_dir: 결과 영상 저장 디렉토리
        conf_threshold: 객체 탐지 신뢰도 임계값
    """
    try:
        # yt-dlp 설치 확인
        try:
            import subprocess
            import tempfile
            
            # 임시 디렉토리 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                # 임시 파일 경로 설정
                temp_video_path = os.path.join(temp_dir, "video.mp4")
                
                print(f"[INFO] 유튜브 영상 '{youtube_url}' 다운로드 중...")
                
                # yt-dlp 명령 실행
                cmd = [
                    "yt-dlp",
                    "-f", "best[ext=mp4]",
                    "-o", temp_video_path,
                    youtube_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"영상 다운로드 실패: {result.stderr}")
                
                if not os.path.exists(temp_video_path):
                    raise Exception("영상 다운로드 후 파일을 찾을 수 없습니다.")
                
                print(f"[INFO] 다운로드 완료: {temp_video_path}")
                
                # 영상 분석
                print("[INFO] 영상 분석 시작...")
                analyze_video(temp_video_path, model_path, output_dir, conf_threshold)
                
                print(f"[INFO] 분석 완료. 결과가 '{output_dir}' 폴더에 저장되었습니다.")
        
        except FileNotFoundError:
            print("[오류] yt-dlp가 설치되어 있지 않습니다. 다음 명령으로 설치하세요:")
            print("pip install yt-dlp")
            return None
            
    except ImportError:
        print("[오류] yt-dlp를 설치해야 합니다. 다음 명령으로 설치하세요:")
        print("pip install yt-dlp")
        return None
    except Exception as e:
        print(f"[오류] 유튜브 영상 처리 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 모델을 사용한 영상 객체 탐지 (로컬 모델)")
    parser.add_argument("--input", type=str, default="videos",
                        help="입력 영상 파일 또는 폴더 경로 (기본값: 'videos' 폴더)")
    parser.add_argument("--youtube", type=str, default=None,
                        help="유튜브 영상 URL (지정하면 --input 무시)")
    parser.add_argument("--model", type=str, default="yolov5s.pt",
                        help="YOLOv5 모델 파일 경로 (기본값: 'yolov5s.pt')")
    parser.add_argument("--output", type=str, default="detected_videos",
                        help="결과 영상 저장 폴더 (기본값: 'detected_videos')")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="객체 탐지 신뢰도 임계값 (기본값: 0.5)")
    
    args = parser.parse_args()
    
    # 유튜브 URL이 제공된 경우
    if args.youtube:
        print(f"[INFO] 유튜브 URL로 실행: {args.youtube}")
        process_youtube_video(args.youtube, args.model, args.output, args.conf)
    
    # 로컬 파일 또는 폴더인 경우
    else:
        # 입력이 폴더인지 파일인지 확인
        if os.path.isdir(args.input):
            process_videos_in_folder(args.input, args.model, args.output, args.conf)
        elif os.path.isfile(args.input):
            analyze_video(args.input, args.model, args.output, args.conf)
        else:
            print(f"[오류] 입력 경로 '{args.input}'가 존재하지 않습니다.")