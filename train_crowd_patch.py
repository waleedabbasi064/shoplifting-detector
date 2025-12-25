from ultralytics import YOLO

# 1. Load your PREVIOUS best model
# This allows us to keep the "retail knowledge" it learned
model = YOLO('runs/detect/yolov12s_thief_detector_12805/weights/best.pt')

print("Starting CrowdHuman Fine-Tuning...")

# 2. Train on the CrowdHuman Dataset
results = model.train(
    # POINT THIS TO YOUR NEW YAML FILE
    data='CrowdHuman.v3-blurhumanfinal.yolov12/data.yaml',
    
    # Training Settings
    epochs=30,             # We only need a short "patch" (20-30 epochs)
    imgsz=1280,            # CRITICAL: Keep high resolution for small objects
    batch=4,               # Lower batch size to be safe with VRAM
    
    # Crowd-Optimized Hyperparameters
    iou=0.65,              # Higher IoU threshold helps separate overlapping people
    box=7.5,               # Increase weight on bounding box accuracy
    
    name='yolov12_crowd_patch'
)

print("Training complete! Use the new best.pt for tracking.")