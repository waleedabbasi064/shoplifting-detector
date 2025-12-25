import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
import os
import time

# ================== MONKEY PATCH FIX START ==================
# This fixes the "AAttn object has no attribute qkv" error
try:
    from ultralytics.nn.modules import block
    original_init = block.AAttn.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.qkv = self.qk

    block.AAttn.__init__ = patched_init
    print("✅ System Ready: Library patched successfully.")
except Exception as e:
    print(f"⚠️ Patch warning: {e}")

# ======================= CONFIGURATION =======================
# 1. Load Models
tracker_model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')
pose_model = YOLO('yolo11n-pose.pt')

# 2. Logic Settings
POCKET_DIST_THRESH = 130      # Hand-to-hip distance (pixels)
SUSPICION_TIME_THRESH = 15    # Frames to wait before turning RED
SCREENSHOT_COOLDOWN = 3       # Seconds between screenshots for the same person

# 3. DANGER ZONE COORDINATES
DANGER_ZONE = np.array([
    [384, 52],    # Top-Left
    [611, 50],    # Top-Right
    [637, 353],   # Bottom-Right
    [386, 346]    # Bottom-Left
], np.int32)

# Create evidence folder
if not os.path.exists('evidence'):
    os.makedirs('evidence')

# Trackers
suspicion_counter = {}
last_capture_time = {}

def is_in_zone(point, polygon):
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

# ======================= VIDEO SETUP =======================
video_path = 'gettyimages-1995820194-640_adpp.mp4' # Ensure this matches your video file
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    'final_system_output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    fps,
    (width, height)
)

print("System Running. Photos will be saved to 'evidence/' folder.")

# ======================= MAIN LOOP =======================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Draw Danger Zone
    cv2.polylines(frame, [DANGER_ZONE], True, (255, 0, 0), 2)
    
    # --- FIX: Use specific [0][0] indices for coordinates ---
    cv2.putText(
        frame, 
        "RESTRICTED AREA", 
        (int(DANGER_ZONE[0][0]), int(DANGER_ZONE[0][1]) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (255, 0, 0), 
        2
    )

    # 1. Track People
    track_results = tracker_model.track(
        frame, persist=True, tracker="reid_tracker.yaml", imgsz=1280, conf=0.15, verbose=False
    )[0]

    # 2. Pose Detection
    pose_results = pose_model(frame, conf=0.3, verbose=False)[0]

    if (track_results.boxes is not None and track_results.boxes.id is not None and pose_results.keypoints is not None):
        track_boxes = track_results.boxes.xyxy.cpu().numpy()
        track_ids = track_results.boxes.id.int().cpu().numpy()
        pose_boxes = pose_results.boxes.xyxy.cpu().numpy()
        keypoints_data = pose_results.keypoints.xy.cpu().numpy()

        for i, t_box in enumerate(track_boxes):
            track_id = track_ids[i]

            # Match pose to tracker
            best_iou = 0
            best_pose_idx = -1
            for j, p_box in enumerate(pose_boxes):
                iou = calculate_iou(t_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pose_idx = j

            if best_iou < 0.5: continue

            kpts = keypoints_data[best_pose_idx]
            feet_x = int((t_box[0] + t_box[2]) / 2)
            feet_y = int(t_box[3])

            # ================= LOGIC =================
            if not is_in_zone((feet_x, feet_y), DANGER_ZONE):
                color = (0, 255, 0)
                label = f"ID:{track_id} Safe"
                suspicion_counter[track_id] = 0
            else:
                dist_L, dist_R = 1000, 1000
                if kpts[9][0] > 0 and kpts[11][0] > 0: dist_L = distance.euclidean(kpts[9], kpts[11])
                if kpts[10][0] > 0 and kpts[12][0] > 0: dist_R = distance.euclidean(kpts[10], kpts[12])

                is_concealing = (dist_L < POCKET_DIST_THRESH) or (dist_R < POCKET_DIST_THRESH)

                if is_concealing:
                    suspicion_counter[track_id] = suspicion_counter.get(track_id, 0) + 1
                else:
                    suspicion_counter[track_id] = max(0, suspicion_counter.get(track_id, 0) - 1)

                if suspicion_counter[track_id] > SUSPICION_TIME_THRESH:
                    color = (0, 0, 255) # RED
                    label = f"ID:{track_id} THEFT!"
                    
                    # --- SCREENSHOT LOGIC ---
                    current_time = time.time()
                    if current_time - last_capture_time.get(track_id, 0) > SCREENSHOT_COOLDOWN:
                        filename = f"evidence/theft_id_{track_id}_{int(current_time)}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 EVIDENCE CAPTURED: {filename}")
                        last_capture_time[track_id] = current_time
                    # ------------------------

                    if dist_L < POCKET_DIST_THRESH:
                        cv2.line(frame, (int(kpts[9][0]), int(kpts[9][1])), (int(kpts[11][0]), int(kpts[11][1])), (0, 0, 255), 3)
                    if dist_R < POCKET_DIST_THRESH:
                        cv2.line(frame, (int(kpts[10][0]), int(kpts[10][1])), (int(kpts[12][0]), int(kpts[12][1])), (0, 0, 255), 3)
                else:
                    color = (0, 165, 255)
                    label = f"ID:{track_id} In Zone"

            x1, y1, x2, y2 = map(int, t_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print("Processing Complete.")