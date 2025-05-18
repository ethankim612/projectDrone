import os
import yaml
import shutil
import random
import argparse
import numpy as np
import ssl
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# SSL 인증서 검증 우회 (보안에 주의)
ssl._create_default_https_context = ssl._create_unverified_context

def prepare_yolov5_dataset(coco_annotation, coco_images, output_dir, split=[0.7, 0.15, 0.15], obstacle_categories=None):
    """
    COCO 데이터셋을 YOLOv5 형식으로 변환
    
    Args:
        coco_annotation (str): COCO 어노테이션 파일 경로
        coco_images (str): COCO 이미지 디렉토리 경로
        output_dir (str): 출력 디렉토리
        split (list): [train, val, test] 비율
        obstacle_categories (list, optional): 장애물로 간주할 카테고리 목록
    """
    from pycocotools.coco import COCO
    
    # COCO 데이터셋 로드
    print(f"COCO 어노테이션 로드 중: {coco_annotation}")
    coco = COCO(coco_annotation)
    
    # 기본 장애물 카테고리 (지정되지 않은 경우)
    if obstacle_categories is None:
        obstacle_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'cat', 'dog'
        ]
    
    print(f"장애물 카테고리: {obstacle_categories}")
    
    # 카테고리 이름으로 ID 매핑
    category_ids = []
    for cat_name in obstacle_categories:
        cat_ids = coco.getCatIds(catNms=[cat_name])
        if cat_ids:
            category_ids.extend(cat_ids)
            print(f"카테고리 '{cat_name}'의 ID: {cat_ids}")
        else:
            print(f"경고: 카테고리 '{cat_name}'을 찾을 수 없습니다.")
    
    # 디렉토리 구조 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    
    # 선택한 카테고리에 해당하는 이미지 ID 가져오기
    ann_ids = coco.getAnnIds(catIds=category_ids)
    all_ann = coco.loadAnns(ann_ids)
    
    # 중복 제거된 이미지 ID 목록
    image_ids = list(set([ann['image_id'] for ann in all_ann]))
    print(f"선택된 이미지 수: {len(image_ids)}")
    
    if len(image_ids) == 0:
        print("오류: 선택된 이미지가 없습니다. 카테고리 이름이 올바른지 확인하세요.")
        return None
    
    # 훈련/검증/테스트 데이터셋 분할
    train_ids, remaining_ids = train_test_split(image_ids, train_size=split[0], random_state=42)
    val_size = split[1] / (split[1] + split[2]) if (split[1] + split[2]) > 0 else 0
    val_ids, test_ids = train_test_split(remaining_ids, train_size=val_size, random_state=42)
    
    print(f"데이터셋 분할: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # 카테고리 ID를 YOLOv5 클래스 인덱스로 매핑
    category_map = {cat_id: i for i, cat_id in enumerate(category_ids)}
    
    # 카테고리 이름 목록
    category_names = [coco.loadCats([cat_id])[0]['name'] for cat_id in category_ids]
    
    # YOLOv5 데이터셋 설정 파일 생성
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(category_ids),
        'names': category_names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"데이터셋 설정 파일 생성: {yaml_path}")
    
    # 이미지와 레이블 변환 함수
    def process_images(image_ids, subset):
        success_count = 0
        for img_id in tqdm(image_ids, desc=f"{subset} 데이터셋 처리 중"):
            try:
                # 이미지 정보 가져오기
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(coco_images, img_info['file_name'])
                
                # 이미지가 존재하지 않으면 건너뛰기
                if not os.path.exists(img_path):
                    print(f"경고: 이미지를 찾을 수 없습니다 - {img_path}")
                    continue
                
                # 이미지 복사
                dest_img_path = os.path.join(output_dir, 'images', subset, img_info['file_name'])
                shutil.copy(img_path, dest_img_path)
                
                # 이미지에 해당하는 어노테이션 가져오기
                ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=category_ids)
                anns = coco.loadAnns(ann_ids)
                
                # YOLOv5 형식으로 레이블 파일 생성
                # (클래스 인덱스, x_center, y_center, width, height) - 모두 정규화된 값
                img_width, img_height = img_info['width'], img_info['height']
                label_path = os.path.join(output_dir, 'labels', subset, 
                                        os.path.splitext(img_info['file_name'])[0] + '.txt')
                
                with open(label_path, 'w') as f:
                    for ann in anns:
                        if ann['category_id'] in category_ids:
                            # COCO 포맷 [x, y, width, height]에서 YOLOv5 포맷으로 변환
                            x, y, w, h = ann['bbox']
                            
                            # 중심점 계산 및 정규화
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # 클래스 인덱스
                            class_idx = category_map[ann['category_id']]
                            
                            # YOLOv5 형식으로 저장
                            f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
                
                success_count += 1
            except Exception as e:
                print(f"이미지 ID {img_id} 처리 중 오류: {str(e)}")
        
        return success_count
    
    # 각 데이터셋 처리
    train_success = process_images(train_ids, 'train')
    val_success = process_images(val_ids, 'val')
    test_success = process_images(test_ids, 'test')
    
    print(f"데이터셋 처리 완료: Train={train_success}, Val={val_success}, Test={test_success}")
    print(f"총 {train_success + val_success + test_success}개 이미지 처리됨")
    return yaml_path

def train_yolov5_model(data_yaml, epochs=50, batch_size=16, img_size=640, weights='yolov5s.pt'):
    """
    YOLOv5 모델 훈련
    
    Args:
        data_yaml (str): 데이터셋 YAML 파일 경로
        epochs (int): 훈련 에포크 수
        batch_size (int): 배치 크기
        img_size (int): 이미지 크기
        weights (str): 초기 가중치 (사전 훈련된 모델)
    """
    # YOLOv5 클론 (필요한 경우)
    if not os.path.exists('yolov5'):
        print("YOLOv5 리포지토리 클론 중...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    
    # 필요한 패키지 설치
    print("필요한 패키지 설치 중...")
    os.system('pip install -r yolov5/requirements.txt')
    
    # 안내 메시지 출력
    print("\n" + "="*80)
    print("중요: 만약 SSL 인증서 오류가 발생한다면, yolov5/utils/general.py 파일을 수정해야 할 수 있습니다.")
    print("518번째 줄 근처의 check_font 함수에서 다음 코드를 찾으세요:")
    print("    torch.hub.download_url_to_file(url, str(file), progress=progress)")
    print("이 줄 위에 다음을 추가하세요:")
    print("    try:")
    print("이 줄 아래에 다음을 추가하세요:")
    print("    except Exception as e:")
    print("        print(f'Warning: Font download failed, using default font. Error: {e}')")
    print("        return")
    print("="*80 + "\n")
    
    # 훈련 명령
    train_command = f"cd yolov5 && python train.py --data {os.path.abspath(data_yaml)} --epochs {epochs} --batch-size {batch_size} --img {img_size} --weights {weights}"
    
    print(f"훈련 명령 실행: {train_command}")
    result = os.system(train_command)
    
    if result == 0:
        print("모델 훈련 완료!")
        print("훈련된 모델은 yolov5/runs/train/ 디렉토리에서 확인할 수 있습니다.")
        
        # 결과 파일 확인
        latest_exp = None
        exp_dirs = [d for d in os.listdir('yolov5/runs/train') if d.startswith('exp')]
        if exp_dirs:
            latest_exp = max(exp_dirs, key=lambda x: int(x.replace('exp', '')) if x != 'exp' else 0)
        
        if latest_exp:
            weights_dir = os.path.join('yolov5/runs/train', latest_exp, 'weights')
            if os.path.exists(os.path.join(weights_dir, 'best.pt')):
                print(f"최상의 모델 가중치: {os.path.join(weights_dir, 'best.pt')}")
            elif os.path.exists(os.path.join(weights_dir, 'last.pt')):
                print(f"마지막 모델 가중치: {os.path.join(weights_dir, 'last.pt')}")
            else:
                print("경고: 모델 가중치 파일을 찾을 수 없습니다.")
    else:
        print("오류: 모델 훈련 중 문제가 발생했습니다.")

def export_yolov5_model(weights_path, format='torchscript'):
    """
    YOLOv5 모델 변환하기
    
    Args:
        weights_path (str): 모델 가중치 파일 경로
        format (str): 변환 형식 ('torchscript', 'onnx', 'coreml' 등)
    """
    if not os.path.exists('yolov5'):
        print("YOLOv5 리포지토리 클론 중...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    
    # 변환 명령
    export_command = f"cd yolov5 && python export.py --weights {weights_path} --include {format}"
    
    print(f"모델 변환 명령 실행: {export_command}")
    os.system(export_command)
    
    print(f"모델 변환 완료! 변환된 모델은 {os.path.splitext(weights_path)[0]}.{format} 파일로 저장되었습니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='COCO 데이터셋으로 YOLOv5 모델 훈련')
    parser.add_argument('--coco_annotation', type=str, default='coco/annotations/instances_val2017.json',
                        help='COCO 어노테이션 파일 경로')
    parser.add_argument('--coco_images', type=str, default='coco/val2017',
                        help='COCO 이미지 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='obstacle_dataset',
                        help='출력 디렉토리')
    parser.add_argument('--categories', type=str, nargs='+',
                        default=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign'],
                        help='감지할 장애물 카테고리 (예: person car)')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='훈련 데이터셋 비율')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='검증 데이터셋 비율')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='테스트 데이터셋 비율')
    parser.add_argument('--epochs', type=int, default=30,
                        help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--img_size', type=int, default=640,
                        help='이미지 크기')
    parser.add_argument('--weights', type=str, default='yolov5s.pt',
                        help='초기 가중치 (사전 훈련된 모델)')
    parser.add_argument('--prepare_only', action='store_true',
                        help='데이터셋만 준비하고 훈련은 하지 않음')
    parser.add_argument('--train_only', action='store_true',
                        help='데이터셋 준비를 건너뛰고 훈련만 수행')
    parser.add_argument('--export', action='store_true',
                        help='훈련 후 모델 내보내기')
    parser.add_argument('--export_format', type=str, default='torchscript',
                        help='내보내기 형식 (torchscript, onnx 등)')
    
    args = parser.parse_args()
    
    if not args.train_only:
        # 데이터셋 준비
        data_yaml = prepare_yolov5_dataset(
            coco_annotation=args.coco_annotation,
            coco_images=args.coco_images,
            output_dir=args.output_dir,
            split=[args.train_split, args.val_split, args.test_split],
            obstacle_categories=args.categories
        )
    else:
        # 기존 데이터셋 사용
        data_yaml = os.path.join(args.output_dir, 'dataset.yaml')
        if not os.path.exists(data_yaml):
            print(f"오류: 데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
            return
    
    if not args.prepare_only and data_yaml:
        # 모델 훈련
        train_yolov5_model(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            weights=args.weights
        )
        
        if args.export:
            # 훈련된 모델 내보내기
            exp_dirs = [d for d in os.listdir('yolov5/runs/train') if d.startswith('exp')]
            if exp_dirs:
                latest_exp = max(exp_dirs, key=lambda x: int(x.replace('exp', '')) if x != 'exp' else 0)
                best_weights = os.path.join('yolov5', 'runs', 'train', latest_exp, 'weights', 'best.pt')
                if os.path.exists(best_weights):
                    export_yolov5_model(best_weights, format=args.export_format)
                else:
                    print(f"오류: 훈련된 모델 가중치 파일을 찾을 수 없습니다: {best_weights}")
            else:
                print("오류: 훈련 결과 디렉토리를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()