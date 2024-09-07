import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fer import FER

# Load the YOLOv8 model from torch hub
model = torch.hub.load('ultralytics/yolov8', 'yolov8x', pretrained=True)

# Initialize the Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Initialize the emotion detector from FER
emotion_detector = FER()

# Open the input video file
cap = cv2.VideoCapture('input_video.mp4')

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_enhanced.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Track the frame count to optimize emotion detection
frame_count = 0

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1  # Increment frame count

    # Use YOLOv8 to perform person detection on the frame
    results = model(frame)
    detections = []

    # Extract bounding box predictions for 'person' class (class id 0)
    for pred in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = pred.tolist()
        if conf > 0.5 and cls == 0:  # Person class (class_id 0)
            detections.append([x1, y1, x2, y2, conf])

    # Convert detections into NumPy format
    detections = np.array(detections)

    # Update the Deep SORT tracker with the detections
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        # Get the bounding box coordinates and track ID
        x1, y1, x2, y2 = track.to_tlbr()
        obj_id = track.track_id

        # Draw bounding box and unique ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Perform emotion detection every 5 frames to optimize processing
        if frame_count % 5 == 0:
            # Crop the face region from the frame
            face = frame[int(y1):int(y2), int(x1):int(x2)]

            # Detect emotion on the cropped face
            if face.size > 0:  # Ensure valid face detection
                emotion, score = emotion_detector.top_emotion(face)

                # Display the emotion label and score on the frame
                cv2.putText(frame, f'{emotion} ({score*100:.1f}%)', (int(x1), int(y2)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (optional for debugging)
    cv2.imshow('frame', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()