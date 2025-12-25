import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
import os
import time
import requests  # New library for Telegram

# ================== USER SETTINGS (EDIT THESE) ==================
TELEGRAM_TOKEN =   "8457525565:AAGRA4pJ64LKXh0gSWBf9p6sYnIpxqjkSYY"
TELEGRAM_CHAT_ID = "7780448652"
USE_TELEGRAM = True # Set to True after pasting your keys above
# ================================================================

# ================== MONKEY PATCH FIX START ==================
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
tracker_model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')
pose_model = YOLO('yolo11n-pose.pt')

POCKET_DIST_THRESH = 130
SUSPICION_TIME_THRESH = 15
SCREENSHOT_COOLDOWN = 5  # Increased cooldown to avoid spamming your phone

# DANGER ZONE
DANGER_ZONE = np.array([
    [384, 52], [611, 50], [637, 353], [386, 346]
], np.int32)

if not os.path.exists('evidence'):
    os.makedirs('evidence')

suspicion_counter = {}
last_capture_time = {}

def is_in_zone(point, polygon):
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def send_telegram_alert(image_path, track_id):
    """Sends the evidence photo to your phone"""
    if not USE_TELEGRAM or "PASTE" in TELEGRAM_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    caption = f"🚨 THEFT DETECTED!\n🆔 Person ID: {track_id}\n📍 Zone: Restricted Area"
    
    try:
        with open(image_path, 'rb') as img:
            files = {'photo': img}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            requests.post(url, files=files, data=data)
            print(f"📲 Alert sent to Telegram for ID {track_id}")
    except Exception as e:
        print(f"⚠️ Failed to send Telegram: {e}")

# ======================= MAIN LOOP =======================
video_path = 'gettyimages-1995820194-640_adpp.mp4'  # CHANGE TO 0 FOR LIVE WEBCAM
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('final_system_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

print("System Running. Monitoring for theft...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Draw Zone
    cv2.polylines(frame, [DANGER_ZONE], True, (255, 0, 0), 2)
    cv2.putText(frame, "RESTRICTED AREA", (int(DANGER_ZONE[0][0]), int(DANGER_ZONE[0][1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Track & Pose
    track_results = tracker_model.track(frame, persist=True, tracker="reid_tracker.yaml", imgsz=1280, conf=0.15, verbose=False)[0]
    pose_results = pose_model(frame, conf=0.3, verbose=False)[0]

    if (track_results.boxes is not None and track_results.boxes.id is not None and pose_results.keypoints is not None):
        track_boxes = track_results.boxes.xyxy.cpu().numpy()
        track_ids = track_results.boxes.id.int().cpu().numpy()
        pose_boxes = pose_results.boxes.xyxy.cpu().numpy()
        kpts_data = pose_results.keypoints.xy.cpu().numpy()

        for i, t_box in enumerate(track_boxes):
            track_id = track_ids[i]
            
            # Match Pose
            best_iou = 0
            best_idx = -1
            for j, p_box in enumerate(pose_boxes):
                iou = calculate_iou(t_box, p_box)
                if iou > best_iou: best_iou, best_idx = iou, j
            
            if best_iou < 0.5: continue

            kpts = kpts_data[best_idx]
            feet = (int((t_box[0]+t_box[2])/2), int(t_box[3]))

            # Logic
            if not is_in_zone(feet, DANGER_ZONE):
                color, label = (0, 255, 0), f"ID:{track_id} Safe"
                suspicion_counter[track_id] = 0
            else:
                dist_L = distance.euclidean(kpts[9], kpts[11]) if kpts[9][0]>0 and kpts[11][0]>0 else 1000
                dist_R = distance.euclidean(kpts[10], kpts[12]) if kpts[10][0]>0 and kpts[12][0]>0 else 1000

                if (dist_L < POCKET_DIST_THRESH) or (dist_R < POCKET_DIST_THRESH):
                    suspicion_counter[track_id] = suspicion_counter.get(track_id, 0) + 1
                else:
                    suspicion_counter[track_id] = max(0, suspicion_counter.get(track_id, 0) - 1)

                if suspicion_counter[track_id] > SUSPICION_TIME_THRESH:
                    color, label = (0, 0, 255), f"ID:{track_id} THEFT!"
                    
                    # Capture & Send Alert
                    curr_time = time.time()
                    if curr_time - last_capture_time.get(track_id, 0) > SCREENSHOT_COOLDOWN:
                        fname = f"evidence/theft_id_{track_id}_{int(curr_time)}.jpg"
                        cv2.imwrite(fname, frame)
                        print(f"📸 Evidence Saved: {fname}")
                        
                        # TRIGGER TELEGRAM
                        send_telegram_alert(fname, track_id)
                        
                        last_capture_time[track_id] = curr_time

                    # Draw red lines
                    if dist_L < POCKET_DIST_THRESH: cv2.line(frame, (int(kpts[9][0]), int(kpts[9][1])), (int(kpts[11][0]), int(kpts[11][1])), (0,0,255), 3)
                    if dist_R < POCKET_DIST_THRESH: cv2.line(frame, (int(kpts[10][0]), int(kpts[10][1])), (int(kpts[12][0]), int(kpts[12][1])), (0,0,255), 3)
                else:
                    color, label = (0, 165, 255), f"ID:{track_id} Suspicious"

            cv2.rectangle(frame, (int(t_box[0]), int(t_box[1])), (int(t_box[2]), int(t_box[3])), color, 2)
            cv2.putText(frame, label, (int(t_box[0]), int(t_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print("Finished.")