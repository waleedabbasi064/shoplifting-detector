import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# --- CONFIGURATION ---
tracker_model = YOLO('runs/detect/yolov12_crowd_patch2/weights/best.pt')
pose_model = YOLO('yolo11n-pose.pt')

# --- LOGIC SETTINGS ---
POCKET_DIST_THRESH = 130
SUSPICION_TIME_THRESH = 15
suspicion_counter = {}


def calculate_iou(box1, box2):
    """Proper IOU calculation"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / float(area1 + area2 - intersection + 1e-6)


# --- VIDEO SETUP ---
video_path = 'gettyimages-1995820194-640_adpp.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('shoplifting_alert.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      fps,
                      (width, height))

print("Starting Shoplifting Logic Engine...")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- RUN TRACKER ---
    track_results = tracker_model.track(
        frame,
        persist=True,
        tracker="reid_tracker.yaml",
        imgsz=1280,
        conf=0.15,
        verbose=False
    )

    # --- RUN POSE ---
    pose_results = pose_model(frame, conf=0.3, verbose=False)

    # --- FIX: ALWAYS TAKE FIRST RESULT ---
    r_track = track_results[0] if isinstance(track_results, list) else track_results
    r_pose = pose_results[0] if isinstance(pose_results, list) else pose_results

    # Skip if no detections
    if r_track.boxes is None or r_track.boxes.id is None:
        out.write(frame)
        continue

    if r_pose.keypoints is None:
        out.write(frame)
        continue

    # Extract tracking data
    track_boxes = r_track.boxes.xyxy.cpu().numpy()
    track_ids = r_track.boxes.id.int().cpu().numpy()

    pose_boxes = r_pose.boxes.xyxy.cpu().numpy()
    keypoints = r_pose.keypoints.xy.cpu().numpy()

    # --- FUSION LOOP ---
    for i, t_box in enumerate(track_boxes):
        current_id = track_ids[i]

        # Match tracking box to closest pose box (using IOU)
        best_iou = 0
        best_idx = -1

        for j, p_box in enumerate(pose_boxes):
            iou = calculate_iou(t_box, p_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou < 0.5:
            continue  # No matching skeleton

        kpts = keypoints[best_idx]

        # Wrist and hip indices:
        # LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_HIP = 11, RIGHT_HIP = 12
        LW, RW, LH, RH = 9, 10, 11, 12

        dist_L = 1000
        dist_R = 1000

        try:
            if kpts[LW][0] > 0 and kpts[LH][0] > 0:
                dist_L = distance.euclidean(kpts[LW], kpts[LH])

            if kpts[RW][0] > 0 and kpts[RH][0] > 0:
                dist_R = distance.euclidean(kpts[RW], kpts[RH])
        except:
            pass

        # Theft condition
        is_near_pocket = (dist_L < POCKET_DIST_THRESH) or (dist_R < POCKET_DIST_THRESH)

        if is_near_pocket:
            suspicion_counter[current_id] = suspicion_counter.get(current_id, 0) + 1
        else:
            suspicion_counter[current_id] = max(0, suspicion_counter.get(current_id, 0) - 1)

        # Drawing
        x1, y1, x2, y2 = map(int, t_box)
        color = (0, 255, 0)
        label = f"ID:{current_id}"

        if suspicion_counter[current_id] > SUSPICION_TIME_THRESH:
            color = (0, 0, 255)
            label = f"ID:{current_id} SUSPICIOUS!"

            # Draw visual proof lines
            if dist_L < POCKET_DIST_THRESH:
                cv2.line(frame,
                         (int(kpts[LW][0]), int(kpts[LW][1])),
                         (int(kpts[LH][0]), int(kpts[LH][1])),
                         (0, 0, 255), 3)

            if dist_R < POCKET_DIST_THRESH:
                cv2.line(frame,
                         (int(kpts[RW][0]), int(kpts[RW][1])),
                         (int(kpts[RH][0]), int(kpts[RH][1])),
                         (0, 0, 255), 3)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()

print("Processing Complete. File saved as 'shoplifting_alert.avi'")

