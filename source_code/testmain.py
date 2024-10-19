import cv2
from object_detection import ObjectDetection

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("your_video_file.mp4")  # Replace with your video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track objects
    frame_with_tracking, (class_ids, scores, boxes) = od.detect_and_track(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame_with_tracking)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()