import cv2
from ultralytics import YOLO
import time
from config import THRESHOLD, CAMERA_SOURCE, DISCORD_WEBHOOK_URL
import os
from datetime import datetime
import requests

def setup_video_capture():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {CAMERA_SOURCE}")
    return cap

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
        
        if video_path:
            # Discord has an 8MB file limit for free accounts
            # Check file size and compress if needed
            if os.path.getsize(video_path) < 8 * 1024 * 1024:  # 8MB in bytes
                files = {
                    "file": ("cat_video.mp4", open(video_path, "rb"), "video/mp4")
                }
            else:
                payload["content"] += "\n(Video too large to attach)"

        requests.post(
            self.webhook_url,
            data=payload,
            files=files
        )

def main():
    model = YOLO('yolov8n.pt')
    cap = setup_video_capture()
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    cat_present = False
    start_time = None
    last_detection_time = None
    DEBOUNCE_THRESHOLD = 2.0
    
    # Add video writer variables
    output_writer = None
    output_frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cat_detected, bbox = detect_cat(frame, model)
            current_time = time.time()
            
            if cat_detected:
                last_detection_time = current_time
                if not cat_present:
                    start_time = current_time
                    cat_present = True
                    print("Cat entered litter box")
                    # Start collecting frames
                    output_frames = []
                
                # Store frame while cat is present
                output_frames.append(frame.copy())
                
            elif cat_present and (current_time - last_detection_time) > DEBOUNCE_THRESHOLD:
                duration = current_time - start_time
                print(f"Cat left after {duration:.1f} seconds")
                
                # Save the video clip
                output_file = None
                if output_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"cat_clips"
                    os.makedirs(output_path, exist_ok=True)
                    
                    height, width = output_frames[0].shape[:2]
                    output_file = os.path.join(output_path, f"cat_visit_{timestamp}.mp4")
                    writer = cv2.VideoWriter(output_file, 
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          30, (width, height))
                    
                    for frame in output_frames:
                        writer.write(frame)
                    writer.release()
                    print(f"Saved video clip to {output_file}")
                
                if duration > THRESHOLD:
                    alert_message = f"⚠️ Alert: Cat spent {duration:.1f} seconds in litter box!"
                    print(alert_message)
                    notifier.send_alert(alert_message, output_file)
                cat_present = False
                output_frames = []
                
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()