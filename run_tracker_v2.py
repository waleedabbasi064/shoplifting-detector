import cv2
from ultralytics import YOLO

# Load your custom model
model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')

print("Starting High-Res Tracker...")
# Run Tracking
results = model.track(
    source='gettyimages-1995820194-640_adpp.mp4',
    
    # CRITICAL FIX 1: Force High Resolution
    # If you trained at 1280, you MUST track at 1280 to see small objects.
    imgsz=1280,
    
    # CRITICAL FIX 2: Lower confidence to catch "blurry" small frames
    conf=0.15,
    
    iou=0.5,
    save=True,
    tracker="sticky_tracker.yaml",
    persist=True
)

print(f"\nTracking complete! Video saved to: {results.save_dir}")