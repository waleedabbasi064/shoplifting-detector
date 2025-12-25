import cv2
from ultralytics import YOLO

# 1. Load your Crowd-Patched Model
model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')

print("Starting ReID Tracker (BoT-SORT)...")

# 2. Run Tracking
results = model.track(
    source='1.mp4',
    imgsz=1280,              # Keep high resolution for small objects
    conf=0.15,               # Keep sensitivity
    # CRITICAL: Use the new BoT-SORT config with ReID enabled
    tracker="reid_tracker.yaml", 
    
    iou=0.5,
    save=True,
    persist=True,
    device='0'               # CRITICAL: ReID needs GPU to run fast
)

print(f"\nTracking complete! Video saved to: {results.save_dir}")