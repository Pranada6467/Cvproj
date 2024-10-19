import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

# Initialize variables
count = 0
tracking_objects = {}
track_id = 0
DETECTION_INTERVAL = 5  # Perform YOLO detection every 5 frames

# KLT tracker parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_frame_gray = None
prev_points = None

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects using YOLO every DETECTION_INTERVAL frames
    if count % DETECTION_INTERVAL == 0:
        (class_ids, scores, boxes) = od.detect(frame)
        
        # Reset tracking objects
        tracking_objects = {}
        
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            
            tracking_objects[track_id] = {'center': (cx, cy), 'box': box}
            track_id += 1

        # Reset KLT points
        prev_points = np.array([obj['center'] for obj in tracking_objects.values()], dtype=np.float32).reshape(-1, 1, 2)

    # Update object positions using KLT
    if prev_frame_gray is not None and prev_points is not None and len(prev_points) > 0:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prev_points, None, **lk_params)
        
        good_new = new_points[status == 1]
        good_old = prev_points[status == 1]

        # Update tracking_objects with KLT results
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            if i < len(tracking_objects):
                object_id = list(tracking_objects.keys())[i]
                tracking_objects[object_id]['center'] = (int(a), int(b))
                
                # Draw the tracks
                cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

        prev_points = good_new.reshape(-1, 1, 2)

    # Draw bounding boxes and object IDs
    for object_id, obj in tracking_objects.items():
        cx, cy = obj['center']
        x, y, w, h = obj['box']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Make object ID clearer
        label = str(object_id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (cx - 10, cy - h - 10), (cx + w + 10, cy), (255, 255, 255), -1)
        cv2.putText(frame, label, (cx - 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Frame", frame)
    
    prev_frame_gray = frame_gray.copy()
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
