import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# --- CONFIGURATION ---
# Load your CUSTOM Crowd-Human patched model for Tracking
detector = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')

# Load the Official YOLO Pose model for Skeletons
pose_model = YOLO('yolo11n-pose.pt')  # Will download automatically

# Thresholds
POCKET_DISTANCE_THRESH = 120  # Pixels (Adjust based on video resolution)
SUSPICION_FRAMES = 15         # How many frames hand must be near pocket to trigger

# Store history to smooth out jitter
suspicious_counter = {}

def get_iou(boxA, boxB):
    # Calculate Intersection over Union to match Tracker Box to Pose Box
    xA = max(boxA, boxB)
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Open Video
cap = cv2.VideoCapture('gettyimages-1995820194-640_adpp.mp4') # Replace with your video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('final_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

print("Starting Behavioral Analysis...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. Run Tracker (Get IDs)
    track_results = detector.track(
        frame, 
        persist=True, 
        tracker="reid_tracker.yaml", 
        conf=0.15, 
        imgsz=1280, 
        verbose=False
    )

    # 2. Run Pose (Get Skeletons)
    pose_results = pose_model(frame, verbose=False, conf=0.3)

    # Process if we have tracks
    if track_results.boxes.id is not None and pose_results.keypoints is not None:
        
        # Get boxes and IDs from Tracker
        track_boxes = track_results.boxes.xyxy.cpu().numpy()
        track_ids = track_results.boxes.id.int().cpu().numpy()
        
        # Get boxes and Keypoints from Pose Model
        pose_boxes = pose_results.boxes.xyxy.cpu().numpy()
        keypoints_data = pose_results.keypoints.xy.cpu().numpy()

        # Match Tracker ID to Pose Skeleton
        for i, t_box in enumerate(track_boxes):
            current_id = track_ids[i]
            best_iou = 0
            best_pose_idx = -1

            # Find which Pose box matches this Tracker box
            for j, p_box in enumerate(pose_boxes):
                iou = get_iou(t_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pose_idx = j
            
            # If we found a matching skeleton for this ID
            if best_iou > 0.5 and best_pose_idx!= -1:
                kpts = keypoints_data[best_pose_idx]
                
                # --- BEHAVIOR LOGIC: Hand-to-Pocket ---
                # Indices: 9=Left Wrist, 10=Right Wrist, 11=Left Hip, 12=Right Hip
                
                # Check Left Hand distance to Left Hip
                if len(kpts) > 12: # Ensure keypoints exist
                    dist_L = distance.euclidean(kpts[4], kpts[5])
                    dist_R = distance.euclidean(kpts[6], kpts[7])
                    
                    # Check if either hand is suspiciously close to hip/pocket
                    if dist_L < POCKET_DISTANCE_THRESH or dist_R < POCKET_DISTANCE_THRESH:
                        suspicious_counter[current_id] = suspicious_counter.get(current_id, 0) + 1
                    else:
                        # Decay counter if hand moves away (prevents false positives from quick swipes)
                        if current_id in suspicious_counter and suspicious_counter[current_id] > 0:
                            suspicious_counter[current_id] -= 1

                    # Determine Status
                    status_color = (0, 255, 0) # Green
                    status_text = f"ID {current_id}: Normal"
                    
                    # Trigger Alert if hand stays in pocket for > X frames
                    if suspicious_counter.get(current_id, 0) > SUSPICION_FRAMES:
                        status_color = (0, 0, 255) # Red
                        status_text = f"ID {current_id}: SUSPICIOUS!"
                        
                        # Draw Skeleton Logic lines
                        if dist_L < POCKET_DISTANCE_THRESH:
                            cv2.line(frame, (int(kpts[4]), int(kpts[4][1])), (int(kpts[5]), int(kpts[5][1])), (0,0,255), 3)
                        if dist_R < POCKET_DISTANCE_THRESH:
                            cv2.line(frame, (int(kpts[6]), int(kpts[6][1])), (int(kpts[7]), int(kpts[7][1])), (0,0,255), 3)

                    # Draw BBox and ID
                    x1, y1, x2, y2 = map(int, t_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                    cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    out.write(frame)

cap.release()
out.release()
print("Done! Check final_output.avi")