from ultralytics import YOLO

print("Starting detector training...")

# Load the COCO-pretrained YOLOv12-S model as our base
model = YOLO('runs/detect/yolov12s_thief_detector_12805/weights/last.pt')

# --- Start Training ---
results = model.train(
    # ================================================================
    #!!! CRITICAL: UPDATE THIS PATH!!!
    # Point this to the 'data.yaml' file inside your Kaggle dataset
    # ================================================================
    resume=True,
    data='Dataset/data.yaml', 
    
    # Use high resolution (1280) for better fidelity
    imgsz=1280,  
    
    # Adjust batch size based on your GPU's VRAM
    # If you get a "CUDA out of memory" error, change this to 4.
    batch=8,     
    
    # Train for 250 epochs
    epochs=250,  
    
    # Specify your GPU
    device=0,    
    
    # Name for the output folder
    name='yolov12s_thief_detector_1280' 
)

# After training, the script will print the final save directory
print("Training complete. Best model saved to:")
print(results.save_dir)