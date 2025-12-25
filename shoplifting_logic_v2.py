import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# --- CONFIGURATION ---
# 1. Your Crowd-Patched Model (The "Tracker")
tracker_model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')

# 2. The Pose Model (The "Skeleton")
pose_model = YOLO('yolo11n-pose.pt') 

# --- LOGIC SETTINGS ---
POCKET_DIST_THRESH = 130   # Pixels: Distance to trigger "touching"
SUSPICION_TIME_THRESH = 15 # Frames: approx 0.5 seconds

# State history
suspicion_counter = {}

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU)"""
    x1 = max(box1, box2)
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1) * (box1[3] - box1[1])
    area2 = (box2[2] - box2) * (box2[3] - box2[1])

    return intersection / float(area1 + area2 - intersection + 1e-6)

# Open Video
video_path = 'gettyimages-1995820194-640_adpp.mp4' 
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('final_output_v3.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

print("Starting Shoplifting Logic Engine v3...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. Run Tracker
    track_results = tracker_model.track(
        frame, 
        persist=True, 
        tracker="reid_tracker.yaml", 
        imgsz=1280,
        conf=0.15,
        verbose=False
    )
    
    # 2. Run Pose
    pose_results = pose_model(frame, verbose=False, conf=0.3)

    # === FIX: EXPLICITLY ACCESS INDEX  ===
    # The models return a list, so we take the first item.
    r_track = track_results
    r_pose = pose_results

    # Check if we have detections AND IDs
    # We check r_track.boxes (not r_track itself)
    if r_track.boxes is not None and r_track.boxes.id is not None and r_pose.keypoints is not None:
        
        # Extract Data
        track_boxes = r_track.boxes.xyxy.cpu().numpy()
        track_ids = r_track.boxes.id.int().cpu().numpy()
        
        pose_boxes = r_pose.boxes.xyxy.cpu().numpy()
        keypoints_data = r_pose.keypoints.xy.cpu().numpy() 

        # --- FUSION LOOP ---
        for i, t_box in enumerate(track_boxes):
            current_id = track_ids[i]
            
            # Match Tracker Box to Pose Box
            best_iou = 0
            best_pose_idx = -1
            for j, p_box in enumerate(pose_boxes):
                iou = calculate_iou(t_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pose_idx = j
            
            # If matched
            if best_iou > 0.5:
                kpts = keypoints_data[best_pose_idx]
                
                # --- THEFT LOGIC ---
                # 9=LeftWrist, 10=RightWrist, 11=LeftHip, 12=RightHip
                
                # Calculate distances (default to 1000 if keypoint missing)
                dist_L = distance.euclidean(kpts[4], kpts[5]) if (kpts[4] > 0 and kpts[5] > 0) else 1000
                dist_R = distance.euclidean(kpts[6], kpts[7]) if (kpts[6] > 0 and kpts[7] > 0) else 1000
                
                is_near_pocket = (dist_L < POCKET_DIST_THRESH) or (dist_R < POCKET_DIST_THRESH)

                if is_near_pocket:
                    suspicion_counter[current_id] = suspicion_counter.get(current_id, 0) + 1
                else:
                    suspicion_counter[current_id] = max(0, suspicion_counter.get(current_id, 0) - 1)

                # Status
                color = (0, 255, 0) # Green
                label = f"ID:{current_id}"

                # TRIGGER ALERT
                if suspicion_counter[current_id] > SUSPICION_TIME_THRESH:
                    color = (0, 0, 255) # Red
                    label = f"ID:{current_id} SUSPICIOUS!"
                    
                    # Draw Logic Lines
                    if dist_L < POCKET_DIST_THRESH:
                        cv2.line(frame, (int(kpts[4]), int(kpts[4][1])), (int(kpts[5]), int(kpts[5][1])), (0,0,255), 3)
                    if dist_R < POCKET_DIST_THRESH:
                        cv2.line(frame, (int(kpts[6]), int(kpts[6][1])), (int(kpts[7]), int(kpts[7][1])), (0,0,255), 3)

                # Draw Box
                x1, y1, x2, y2 = map(int, t_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print("Processing Complete. Download 'final_output_v3.avi'")