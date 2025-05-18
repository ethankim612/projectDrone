import os
import cv2
import numpy as np
import torch
import json
from pycocotools.coco import COCO
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

class COCOObstacleDetector:
    def __init__(self, coco_annotation_path, coco_images_dir, model_type='yolo', model_path=None):
        """
        COCO 데이터셋을 사용한 장애물 감지기 초기화
        
        Args:
            coco_annotation_path (str): COCO 어노테이션 파일 경로 (예: 'coco/annotations/instances_val2017.json')
            coco_images_dir (str): COCO 이미지 디렉토리 경로 (예: 'coco/val2017/')
            model_type (str): 'yolo' 또는 'faster_rcnn' 또는 'custom'
            model_path (str): 사용자 정의 모델 경로 (예: 'yolov5/runs/train/exp2/weights/best.pt')
        """
        # COCO 데이터셋 로드
        print(f"COCO 어노테이션 로드 중: {coco_annotation_path}")
        self.coco = COCO(coco_annotation_path)
        self.images_dir = coco_images_dir
        
        # 장애물로 간주할 카테고리 ID 목록
        # COCO 카테고리에서 장애물로 볼 수 있는 객체들 (차량, 사람, 동물 등)
        self.obstacle_cat_ids = [
            1,  # person
            2,  # bicycle
            3,  # car
            4,  # motorcycle
            6,  # bus
            8,  # truck
            17, # cat
            18, # dog
            13, # stop sign
            10  # traffic light
        ]
        
        # 카테고리 ID와 이름 매핑
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        print(f"감지 대상 카테고리: {[self.categories[cat_id] for cat_id in self.obstacle_cat_ids if cat_id in self.categories]}")
        
        # 모델 로드
        print(f"{model_type} 모델 로드 중...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 장치: {self.device}")
        
        # 인터넷 다운로드 비활성화
        os.environ['TORCH_HOME'] = './torch_home'
        os.environ['TORCH_OFFLINE'] = '1'  # 오프라인 모드 활성화
        
        # 모델 초기화
        if model_path and os.path.exists(model_path):
            # 학습된 커스텀 모델 사용
            print(f"학습된 모델 로드 중: {model_path}")
            try:
                # 로컬 YOLOv5 리포지토리가 있는 경우
                if os.path.exists('yolov5'):
                    self.model = self.load_local_model(model_path)
                else:
                    # 그렇지 않으면 torch.load 사용
                    self.model = self.load_torch_model(model_path)
            except Exception as e:
                print(f"모델 로드 중 오류 발생: {e}")
                raise
        elif model_type == 'yolo':
            # 오프라인 환경에서만 YOLOv5 모델 로드 시도
            try:
                self.model = self.load_yolo_offline()
            except Exception as e:
                print(f"기본 YOLOv5 모델 로드 실패: {e}")
                print("기본 YOLOv5 모델을 찾을 수 없습니다. 학습된 모델 경로를 지정하세요.")
                raise
        elif model_type == 'faster_rcnn':
            # Faster R-CNN 모델 로드
            try:
                from torchvision.models import detection
                self.model = detection.fasterrcnn_resnet50_fpn(pretrained=False)  # 온라인 다운로드 비활성화
                # 로컬 모델 파일 검사
                rcnn_path = './fasterrcnn_resnet50_fpn.pth'
                if os.path.exists(rcnn_path):
                    self.model.load_state_dict(torch.load(rcnn_path, map_location=self.device))
                    print(f"Faster R-CNN 로컬 모델 로드 완료: {rcnn_path}")
                else:
                    print("Faster R-CNN 모델 파일을 찾을 수 없습니다.")
                    print("미리 다운로드된 모델 파일이 필요합니다.")
                    raise FileNotFoundError(f"Faster R-CNN 모델 파일을 찾을 수 없습니다: {rcnn_path}")
                
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Faster R-CNN 모델 로드 실패: {e}")
                raise
    
    def load_local_model(self, model_path):
        """로컬 YOLOv5 리포지토리를 사용하여 모델 로드"""
        # torch.hub.load 사용 시 인터넷 연결 시도 방지
        torch.hub.set_dir('./torch_hub')
        # source='local' 옵션으로 로컬 파일만 사용
        model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')
        model.to(self.device)
        print(f"로컬 YOLOv5 리포지토리를 사용하여 모델 로드 완료: {model_path}")
        return model
    
    def load_torch_model(self, model_path):
        """torch.load를 사용하여 모델 로드"""
        model = torch.load(model_path, map_location=self.device)
        if isinstance(model, dict):
            model = model['model']
        model.to(self.device)
        print(f"torch.load를 사용하여 모델 로드 완료: {model_path}")
        return model
    
    def load_yolo_offline(self):
        """오프라인 환경에서 YOLOv5 모델 로드 시도"""
        # 로컬 모델 파일 목록
        model_files = [
            './yolov5s.pt',                                              # 현재 디렉토리
            './weights/yolov5s.pt',                                      # weights 폴더
            './yolov5/yolov5s.pt',                                       # yolov5 폴더
            os.path.expanduser('~/.cache/torch/hub/checkpoints/yolov5s.pt'),  # 캐시 디렉토리
            os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master/yolov5s.pt'),  # 캐시 디렉토리 2
            os.path.expanduser('~/yolov5s.pt'),                          # 홈 디렉토리
        ]
        
        # 모델 파일 검색
        model_path = None
        for path in model_files:
            if os.path.exists(path):
                model_path = path
                print(f"YOLOv5 모델 파일 발견: {path}")
                break
        
        # 모델 파일을 찾지 못한 경우
        if model_path is None:
            # 로컬 YOLOv5 리포지토리가 있는지 확인
            if os.path.exists('./yolov5'):
                print("로컬 YOLOv5 리포지토리 발견. 모델 파일을 다운로드한 후 오프라인으로 실행하세요.")
                print("먼저 다음 명령을 실행하여 모델 파일을 다운로드하세요:")
                print("  python -c \"import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'yolov5s.pt')\"")
                raise FileNotFoundError("YOLOv5 모델 파일을 찾을 수 없습니다.")
            else:
                print("로컬 YOLOv5 리포지토리를 찾을 수 없습니다.")
                print("먼저 다음 명령을 실행하여 로컬에 리포지토리를 클론하세요:")
                print("  git clone https://github.com/ultralytics/yolov5.git")
                print("  cd yolov5")
                print("  pip install -r requirements.txt")
                raise FileNotFoundError("YOLOv5 모델 파일과 리포지토리를 찾을 수 없습니다.")
        
        # 모델 로드 방식 결정
        if os.path.exists('./yolov5'):
            # YOLOv5 리포지토리가 있는 경우 torch.hub.load 사용
            try:
                model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')
                print("YOLOv5 리포지토리를 사용하여 모델 로드 완료")
                return model
            except Exception as e:
                print(f"torch.hub.load 실패: {e}")
                # 실패할 경우 torch.load로 대체
                return self.load_torch_model(model_path)
        else:
            # 리포지토리가 없는 경우 직접 모델 파일 로드
            return self.load_torch_model(model_path)
    
    def get_obstacle_images(self, num_images=5):
        """
        장애물이 포함된 이미지 ID 목록 가져오기
        
        Args:
            num_images (int): 가져올 이미지 수
            
        Returns:
            list: 이미지 ID 목록
        """
        # 장애물 카테고리에 해당하는 이미지 ID 가져오기
        img_ids = []
        for cat_id in self.obstacle_cat_ids:
            if cat_id in self.categories:  # 카테고리 ID가 존재하는지 확인
                ann_ids = self.coco.getAnnIds(catIds=[cat_id])
                img_ids.extend([ann['image_id'] for ann in self.coco.loadAnns(ann_ids)])
        
        # 중복 제거 및 제한
        img_ids = list(set(img_ids))[:num_images]
        print(f"{len(img_ids)}개의 이미지 ID 선택됨")
        return img_ids
    
    def load_image_by_id(self, img_id):
        """
        이미지 ID로 이미지 로드
        
        Args:
            img_id (int): 이미지 ID
            
        Returns:
            tuple: (이미지, 파일명)
        """
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"이미지를 찾을 수 없습니다: {img_path}")
        
        # BGR -> RGB 변환 (YOLO는 RGB 형식을 사용)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_info['file_name']
    
    def get_ground_truth(self, img_id):
        """
        이미지의 실제 어노테이션(ground truth) 가져오기
        
        Args:
            img_id (int): 이미지 ID
            
        Returns:
            list: 장애물 어노테이션 목록
        """
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        obstacles = []
        for ann in anns:
            if ann['category_id'] in self.obstacle_cat_ids:
                # COCO 형식의 bbox [x, y, width, height]를 [x1, y1, x2, y2]로 변환
                x, y, w, h = ann['bbox']
                obstacles.append({
                    'category_id': ann['category_id'],
                    'category_name': self.categories[ann['category_id']],
                    'bbox': [int(x), int(y), int(x + w), int(y + h)]
                })
        
        return obstacles
    
    def detect_obstacles(self, img):
        """
        이미지에서 장애물 감지
        
        Args:
            img (numpy.ndarray): 감지할 이미지 (RGB 형식)
            
        Returns:
            tuple: (표시용 이미지, 감지된 장애물 목록)
        """
        # YOLOv5 사용 시
        if hasattr(self.model, 'names') or hasattr(self.model, 'predict'):
            # YOLOv5 모델로 객체 감지
            results = self.model(img)
            
            # 결과 처리 - 원본 이미지 반환 (렌더링된 이미지 대신)
            result_img = img.copy()
            
            # 감지된 객체 정보 추출
            obstacles = []
            for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
                x1, y1, x2, y2 = [int(coord) for coord in box]
                class_id = int(cls)
                class_name = self.model.names[class_id]
                confidence = float(conf)
                
                obstacles.append({
                    'category_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Faster R-CNN 사용 시
        else:
            img_tensor = F.to_tensor(Image.fromarray(img)).to(self.device)
            
            with torch.no_grad():
                predictions = self.model([img_tensor])
                
            # 결과 처리
            result_img = img.copy()
            obstacles = []
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if score > 0.5:  # 신뢰도 임계값
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = f"Class {label}"  # COCO 클래스로 매핑 필요
                    
                    obstacles.append({
                        'category_name': class_name,
                        'confidence': float(score),
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return result_img, obstacles
    
    def visualize_detection(self, img, obstacles, gt_obstacles=None, show_text_twice=False):
        """
        감지 결과를 시각화
        
        Args:
            img (numpy.ndarray): 원본 이미지
            obstacles (list): 감지된 장애물 목록
            gt_obstacles (list, optional): 실제 장애물 목록
            show_text_twice (bool): 텍스트를 두 번 표시할지 여부 (기본값: False)
            
        Returns:
            numpy.ndarray: 시각화된 이미지
        """
        vis_img = img.copy()
        
        # 감지된 장애물 표시 (파란색 배경에 흰색 텍스트)
        for obj in obstacles:
            x1, y1, x2, y2 = obj['bbox']
            
            # 경계 상자 그리기 (빨간색)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 텍스트 한 번만 표시 (파란색 배경에 흰색 텍스트)
            if 'confidence' in obj:
                label = f"{obj['category_name']} {obj['confidence']:.2f}"
                # 텍스트 크기 계산
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 텍스트 배경 (파란색)
                cv2.rectangle(vis_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
                
                # 텍스트 (흰색)
                cv2.putText(vis_img, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 실제 장애물 표시 (녹색)
        if gt_obstacles:
            for obj in gt_obstacles:
                x1, y1, x2, y2 = obj['bbox']
                
                # 경계 상자 그리기 (녹색)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Ground Truth 텍스트 (녹색 배경에 흰색 텍스트)
                label = f"GT: {obj['category_name']}"
                
                # 텍스트 크기 계산
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 텍스트 배경 (녹색)
                cv2.rectangle(vis_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                
                # 텍스트 (흰색)
                cv2.putText(vis_img, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_img
    
    def evaluate_on_dataset(self, num_samples=10, output_dir='results'):
        """
        데이터셋에서 모델 평가
        
        Args:
            num_samples (int): 평가할 이미지 수
            output_dir (str): 결과 저장 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        img_ids = self.get_obstacle_images(num_samples)
        
        results = []
        for i, img_id in enumerate(img_ids):
            try:
                print(f"이미지 {i+1}/{len(img_ids)} 처리 중 (ID: {img_id})...")
                img, file_name = self.load_image_by_id(img_id)
                gt_obstacles = self.get_ground_truth(img_id)
                
                # 원본 이미지 저장
                original_output_path = os.path.join(output_dir, f"original_{i}_{file_name}")
                cv2.imwrite(original_output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # 장애물 감지 실행
                result_img, detected_obstacles = self.detect_obstacles(img)
                
                # 결과 시각화
                vis_img = self.visualize_detection(img, detected_obstacles, gt_obstacles, show_text_twice=False)
                
                # 결과 저장
                output_path = os.path.join(output_dir, f"result_{i}_{file_name}")
                cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                # 결과 기록
                results.append({
                    'image_id': img_id,
                    'file_name': file_name,
                    'original_image': f"original_{i}_{file_name}",
                    'result_image': f"result_{i}_{file_name}",
                    'ground_truth': gt_obstacles,
                    'detections': detected_obstacles
                })
                
                print(f"이미지 {i+1}/{len(img_ids)} 처리 완료: {file_name}")
                
            except Exception as e:
                print(f"이미지 ID {img_id} 처리 중 오류: {e}")
        
        # 결과를 JSON으로 저장
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"평가 완료. 결과가 {output_dir}에 저장되었습니다.")
        print(f"- 원본 이미지: original_X_filename.jpg")
        print(f"- 결과 이미지: result_X_filename.jpg")
    
    def detect_from_video(self, video_path=0, output_path=None, conf_threshold=0.5):
        """
        비디오에서 장애물 감지
        
        Args:
            video_path: 비디오 파일 경로 또는 카메라 인덱스 (0은 웹캠)
            output_path: 결과 비디오 저장 경로 (None이면 저장하지 않음)
            conf_threshold: 객체 감지 신뢰도 임계값
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"비디오를 열 수 없습니다: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"비디오 크기: {width}x{height}, FPS: {fps}")
        
        # 비디오 저장 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 원본 비디오도 저장
            original_output_path = output_path.replace('.avi', '_original.avi')
            original_out = cv2.VideoWriter(original_output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 원본 프레임 저장
                if output_path:
                    original_out.write(frame)
                
                # BGR -> RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 장애물 감지
                _, obstacles = self.detect_obstacles(rgb_frame)
                
                # 결과 시각화
                vis_frame = self.visualize_detection(rgb_frame, obstacles, show_text_twice=False)
                
                # RGB -> BGR 변환 (OpenCV 표시용)
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                
                # 결과 표시
                cv2.imshow('Obstacle Detection', vis_frame_bgr)
                
                # 결과 저장
                if output_path:
                    out.write(vis_frame_bgr)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"처리된 프레임: {frame_count}")
                
                # q 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"비디오 처리 중 오류: {e}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
                original_out.release()
                print(f"비디오 처리 완료. 총 {frame_count}개 프레임 처리됨.")
                print(f"결과가 저장됨:")
                print(f"- 원본 비디오: {original_output_path}")
                print(f"- 결과 비디오: {output_path}")
            cv2.destroyAllWindows()

# 단독 실행 테스트
if __name__ == "__main__":
    import argparse
    
    # 시작 시 오프라인 모드 설정
    os.environ['TORCH_HOME'] = './torch_home'
    os.environ['TORCH_OFFLINE'] = '1'  # 오프라인 모드 활성화
    
    parser = argparse.ArgumentParser(description='COCO 장애물 감지 테스트')
    parser.add_argument('--coco_annotation', type=str, default='coco/annotations/instances_val2017.json', 
                      help='COCO 어노테이션 파일 경로')
    parser.add_argument('--coco_images', type=str, default='coco/val2017', 
                      help='COCO 이미지 디렉토리 경로')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['yolo', 'faster_rcnn', 'custom'],
                      help='사용할 모델 타입')
    parser.add_argument('--model_path', type=str, default=None, 
                      help='학습된 모델 파일 경로 (예: yolov5/runs/train/exp2/weights/best.pt)')
    parser.add_argument('--num_samples', type=int, default=10, 
                      help='테스트할 샘플 수')
    parser.add_argument('--output_dir', type=str, default='obstacle_detection_results', 
                      help='결과 저장 디렉토리')
    parser.add_argument('--video', type=str, default=None, 
                      help='비디오 파일 경로 (지정하면 비디오 테스트 모드)')
    parser.add_argument('--webcam', action='store_true', 
                      help='웹캠 사용 (지정하면 웹캠 테스트 모드)')
    
    args = parser.parse_args()
    
    # 감지기 초기화
    try:
        detector = COCOObstacleDetector(
            args.coco_annotation, 
            args.coco_images, 
            model_type=args.model_type,
            model_path=args.model_path
        )
        
        # 비디오 또는 웹캠 모드
        if args.video:
            detector.detect_from_video(args.video, f"{args.output_dir}/video_result.avi")
        elif args.webcam:
            detector.detect_from_video(0, f"{args.output_dir}/webcam_result.avi")
        else:
            # 데이터셋에서 모델 평가
            detector.evaluate_on_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        
        # YOLOv5 리포지토리가 없는 경우 가이드 제공
        if "YOLOv5 모델을 찾을 수 없습니다" in str(e):
            print("\n=== 해결 방법 ===")
            print("1. YOLOv5 리포지토리 클론:")
            print("   git clone https://github.com/ultralytics/yolov5.git")
            print("   cd yolov5")
            print("   pip install -r requirements.txt")
            print("\n2. YOLOv5 모델 파일 다운로드 (인터넷 연결 필요 - 한 번만 실행):")
            print("   python -c \"import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'yolov5s.pt')\"")
            print("\n3. 학습된 모델 파일 경로 지정:")
            print("   python coco_obstacle_detection.py --model_path='yolov5/runs/train/exp2/weights/best.pt' --model_type=custom")