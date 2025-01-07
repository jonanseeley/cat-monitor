import cv2
from ultralytics import YOLO
import time
from config import THRESHOLD, CAMERA_SOURCE1, CAMERA_SOURCE2, CAMERA_SOURCE3, DISCORD_WEBHOOK_URL
import os
from datetime import datetime
import requests
from typing import Dict, List, Tuple

def detect_cat(frame, model):
    results = model(frame, verbose=False)
    # Check if any detected object is a cat (class 15 in COCO dataset)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 15:  # cat class
                return True, box.xyxy[0].tolist()
    return False, None

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_alert(self, message, video_path=None):
        payload = {"content": message}
        files = {}
        
        if video_path and os.path.getsize(video_path) < 8 * 1024 * 1024:
            # Read the file content before making the request
            with open(video_path, "rb") as video_file:
                video_content = video_file.read()
                files = {
                    "file": ("cat_video.mp4", video_content, "video/mp4")
                }
            
        requests.post(
            self.webhook_url,
            data=payload,
            files=files
        )

class CameraMonitor:
    def __init__(self, camera_sources: Dict[str, str], webhook_url: str):
        self.camera_sources = camera_sources
        self.captures = {}
        self.cat_states = {}  # Track cat presence for each camera
        self.start_times = {}
        self.last_detection_times = {}
        self.output_frames = {}
        self.notifier = DiscordNotifier(webhook_url)
        self.model = YOLO('yolov8n.pt')
        self.DEBOUNCE_THRESHOLD = 4.0
        self.DETECTION_INTERVAL = 1.0  # seconds between detection checks
        self.last_check_times = {camera_id: 0 for camera_id in camera_sources}
        self.frame_skip = 150  # Process every nth frame
        self.frame_count = {camera_id: 0 for camera_id in camera_sources}
        
        # Initialize all cameras
        for camera_id, source in camera_sources.items():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_id}: {source}")
            self.captures[camera_id] = cap
            self.cat_states[camera_id] = False
            self.start_times[camera_id] = None
            self.last_detection_times[camera_id] = None
            self.output_frames[camera_id] = []
        
        # Add error counters for each camera
        self.error_counts = {camera_id: 0 for camera_id in camera_sources}
        self.MAX_ERRORS = 10  # Maximum number of consecutive errors before reconnecting
        
    def _reconnect_camera(self, camera_id: str) -> None:
        """Attempt to reconnect to a camera after errors"""
        print(f"Attempting to reconnect to camera {camera_id}...")
        if self.captures[camera_id].isOpened():
            self.captures[camera_id].release()
        
        self.captures[camera_id] = cv2.VideoCapture(self.camera_sources[camera_id])
        self.error_counts[camera_id] = 0
        
        # Set camera properties that might help with stability
        self.captures[camera_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        
    def process_frame(self, camera_id: str, frame) -> None:
        self.frame_count[camera_id] += 1
        current_time = time.time()
        
        # Skip frames for performance unless we're tracking a cat
        if not self.cat_states[camera_id] and self.frame_count[camera_id] % self.frame_skip != 0:
            return
            
        # Always append frame if we're currently tracking a cat
        if self.cat_states[camera_id]:
            self.output_frames[camera_id].append(frame.copy())
        
        # Only run detection if enough time has passed since last check
        if current_time - self.last_check_times[camera_id] >= self.DETECTION_INTERVAL:
            self.last_check_times[camera_id] = current_time
            cat_detected, bbox = detect_cat(frame, self.model)
            
            if cat_detected:
                self.last_detection_times[camera_id] = current_time
                if not self.cat_states[camera_id]:
                    self.start_times[camera_id] = current_time
                    self.cat_states[camera_id] = True
                    print(f"Cat entered litter box on camera {camera_id}")
                    self.output_frames[camera_id] = [frame.copy()]  # Start with current frame
            
            elif (self.cat_states[camera_id] and 
                  (current_time - self.last_detection_times[camera_id]) > self.DEBOUNCE_THRESHOLD):
                self._handle_cat_exit(camera_id, current_time)

    def _handle_cat_exit(self, camera_id: str, current_time: float) -> None:
        duration = current_time - self.start_times[camera_id]
        print(f"Cat left after {duration:.1f} seconds on camera {camera_id}")
        
        output_file = None
        if self.output_frames[camera_id]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cat_clips/{camera_id}"
            os.makedirs(output_path, exist_ok=True)
            
            frames = self.output_frames[camera_id]
            height, width = frames[0].shape[:2]
            output_file = os.path.join(output_path, f"cat_visit_{timestamp}.mp4")
            
            # Use more efficient codec and preset
            writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 codec
                
15, (width, height)
            )
            
            # Write frames in chunks
            for i in range(0, len(frames), 10):
                chunk = frames[i:i+10]
                writer.write(chunk)
            writer.release()
            print(f"Saved video clip to {output_file}")
        
        if duration > THRESHOLD:
            alert_message = f"⚠️ Alert: Cat spent {duration:.1f} seconds in litter box on camera {camera_id}!"
            print(alert_message)
            self.notifier.send_alert(alert_message, output_file)
        
        self.cat_states[camera_id] = False
        self.output_frames[camera_id] = []

    def run(self):
        print("Starting monitoring...")
        try:
            while True:
                for camera_id, cap in self.captures.items():
                    ret, frame = cap.read()
                    if not ret:
                        self.error_counts[camera_id] += 1
                        print(f"Error reading from camera {camera_id} ({self.error_counts[camera_id]}/{self.MAX_ERRORS})")
                        
                        if self.error_counts[camera_id] >= self.MAX_ERRORS:
                            self._reconnect_camera(camera_id)
                        continue
                    
                    # Reset error count on successful frame read
                    self.error_counts[camera_id] = 0
                    self.process_frame(camera_id, frame)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        finally:
            for cap in self.captures.values():
                cap.release()
            cv2.destroyAllWindows()

def main():
    # Example camera sources
    camera_sources = {
        "box1": CAMERA_SOURCE1,
        "box2": CAMERA_SOURCE2,
        "box3": CAMERA_SOURCE3
    }
    
    monitor = CameraMonitor(camera_sources, DISCORD_WEBHOOK_URL)
    monitor.run()

if __name__ == "__main__":
    main()