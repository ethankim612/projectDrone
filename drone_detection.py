import cv2
import time
import os
import torch
import numpy as np
from datetime import datetime
from threading import Thread
import queue

class RTMPStreamProcessor:
    def __init__(self, rtmp_url, model_path):
        self.rtmp_url = rtmp_url
        self.model_path = model_path
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False
        
        self.save_dir = "stream_captures"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.recording = False
        self.video_writer = None
        
        self.detection_enabled = True
        self.detection_confidence = 0.5
        self.process_every_n_frames = 2
        self.counter = 0
        
        self.display_width = 800
        self.display_height = 600
        self.inference_size = 416
        
        self.processed_frame = None

    def start_capture(self):
        print(f"Attempting to connect to RTMP stream: {self.rtmp_url}")
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;512000|max_delay;50|fflags;nobuffer|flags;low_delay"
        
        self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not self.cap.isOpened():
            print("Error: Cannot open RTMP stream.")
            return False
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"RTMP stream connection successful!")
        print(f"Video info: {self.width}x{self.height}, {self.fps}fps")
        
        self.capture_thread = Thread(target=self._capture_frames, name="CaptureThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def load_model(self):
        print(f"Loading trained YOLOv5 object detection model: {self.model_path}")
        
        try:
            self.model = torch.hub.load('yolov5', 'custom', path=self.model_path, source='local')
            
            self.model.conf = self.detection_confidence
            self.model.iou = 0.45
            self.model.classes = None
            self.model.img = self.inference_size
            
            self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
            
            print(f"YOLOv5 model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            return False
    
    def _capture_frames(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print("Capture connection lost. Attempting to reconnect...")
                self.cap.release()
                time.sleep(1)
                
                self.cap = cv2.VideoCapture(self.rtmp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                if not self.cap.isOpened():
                    time.sleep(2)
                    continue
            
            ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.01)
                continue
            
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            time.sleep(0.001)
    
    def process_frames(self):
        cv2.namedWindow('RTMP Stream', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('RTMP Stream', self.display_width, self.display_height)
        
        while not self.stopped:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = self.frame_queue.get_nowait()
                
                self.counter += 1
                process_this_frame = (self.counter % self.process_every_n_frames == 0)
                
                if self.processed_frame is None or process_this_frame:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, current_time, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if self.detection_enabled and process_this_frame:
                        self.model.conf = self.detection_confidence
                        
                        results = self.model(frame)
                        
                        detections = results.pandas().xyxy[0]
                        
                        detected_objects = {}
                        
                        for _, detection in detections.iterrows():
                            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                            
                            label = detection['name']
                            confidence = detection['confidence']
                            
                            if label in detected_objects:
                                detected_objects[label] += 1
                            else:
                                detected_objects[label] = 1
                            
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            text = f"{label}: {confidence:.2f}"
                            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        offset_y = 60
                        for label, count in detected_objects.items():
                            cv2.putText(frame, f"{label}: {count}", (10, offset_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            offset_y += 25
                        
                        cv2.putText(frame, "Detection: ON", (self.width - 150, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif not self.detection_enabled:
                        cv2.putText(frame, "Detection: OFF", (self.width - 150, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if self.recording:
                        cv2.putText(frame, "REC", (self.width - 70, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    self.processed_frame = frame
                
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(self.processed_frame)
                
                cv2.imshow('RTMP Stream', self.processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if not self.handle_key_events(key):
                    break
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                time.sleep(0.1)
        
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
    
    def handle_key_events(self, key):
        if key == ord('q'):
            return False
            
        elif key == ord('s'):
            self.save_snapshot()
            
        elif key == ord('r'):
            self.toggle_recording()
        
        elif key == ord('d'):
            self.detection_enabled = not self.detection_enabled
            print(f"Object detection: {'enabled' if self.detection_enabled else 'disabled'}")
        
        elif key == ord('+') or key == ord('='):
            self.detection_confidence = min(self.detection_confidence + 0.05, 0.95)
            print(f"Detection confidence threshold: {self.detection_confidence:.2f}")
        
        elif key == ord('-'):
            self.detection_confidence = max(self.detection_confidence - 0.05, 0.05)
            print(f"Detection confidence threshold: {self.detection_confidence:.2f}")
            
        elif key == ord('f'):
            self.process_every_n_frames = min(self.process_every_n_frames + 1, 10)
            print(f"Processing every {self.process_every_n_frames}th frame")
            
        elif key == ord('g'):
            self.process_every_n_frames = max(self.process_every_n_frames - 1, 1)
            print(f"Processing every {self.process_every_n_frames}th frame")
        
        return True
    
    def save_snapshot(self):
        if self.processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"{self.save_dir}/snapshot_{timestamp}.jpg"
            cv2.imwrite(img_path, self.processed_frame)
            print(f"Snapshot saved: {img_path}")
    
    def toggle_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"{self.save_dir}/video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, 
                                              (self.width, self.height))
            self.recording = True
            print(f"Recording started: {video_path}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("Recording stopped")
    
    def stop(self):
        self.stopped = True
        
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        print("Program has terminated.")

def main():
    rtmp_url = "rtmp://11.16.45.11:1935/live/test"
    
    model_paths = [
        "./yolov5/yolov5s.pt",
        "./yolov5/yolov5su.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Error: YOLOv5 model file not found.")
        print("Checked paths:", model_paths)
        return
    
    try:
        processor = RTMPStreamProcessor(rtmp_url, model_path)
        
        if not processor.start_capture():
            print("Failed to connect to stream.")
            return
        
        if not processor.load_model():
            print("Failed to load model.")
            processor.stop()
            return
        
        processor.process_frames()
    
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        if 'processor' in locals():
            processor.stop()

if __name__ == "__main__":
    main()