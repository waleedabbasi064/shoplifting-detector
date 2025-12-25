from ultralytics import YOLO
import os

# 1. Load your custom-trained model (The "Gold" file)
# Make sure this path matches exactly where your training finished
model_path = 'runs/detect/yolov12s_thief_detector_12805/weights/best.pt'
model = YOLO(model_path)

# 2. Define the source to test
# OPTION A: Test on a single image from your dataset's test set
# (Replace this filename with an actual file from your Dataset/test/images/ folder)
source = 'gettyimages-1995820194-640_adpp.mp4' 


# OPTION B: Test on a video (Upload a video to your folder first)
# source = 'shoplifting_test_video.mp4'

# Check if source exists before running
if not os.path.exists(source):
    print(f"Error: Source file '{source}' not found. Please check the path.")
else:
    print(f"Testing model on: {source}...")

    # 3. Run Inference
    # conf=0.25 : Only show detections with >25% confidence
    # save=True : Save the output video/image to disk
    # imgsz=1280: CRITICAL - Use the same high resolution we trained with!
    results = model.predict(
        source=source, 
        save=True, 
        imgsz=1280, 
        conf=0.25,
        iou=0.45  # NMS threshold to remove duplicate boxes
    )

    print("\nTest complete!")
    print(f"Results saved to: {results.save_dir}")