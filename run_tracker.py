import cv2
from ultralytics import YOLO

# 1. Load your custom-trained model from Part 1
# This is the specific path where your training finished successfully
model = YOLO('runs/detect/yolov12s_thief_detector_12805/weights/best.pt')

print("Starting Object Tracking...")

# 2. Run the Tracker
# tracker="botsort.yaml" : Uses the BoT-SORT algorithm (better for occlusion than ByteTrack)
# persist=True           : CRITICAL. Forces the system to remember IDs between frames.
results = model.track(
    source='gettyimages-1995820194-640_adpp.mp4',       # The video file you uploaded
    conf=0.25,            # Confidence threshold (ignore weak detections)
    iou=0.5,              # NMS threshold (remove duplicate boxes)
    save=True,            # Save the output video with IDs drawn
    tracker="botsort.yaml", 
    persist=True,
    show=False            # Don't pop up a window (since you are on a remote server)
)

# 3. Print the location of the saved video
# We access results because 'model.track' returns a list
print(f"\nTracking complete! Video saved to: {results.save_dir}")