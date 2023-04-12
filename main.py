import cv2
import numpy as np

# Load the pre-trained object detection model
model = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the video stream
video_stream = cv2.VideoCapture(0)

# Initialize the tracker
tracker = cv2.TrackerKCF_create()

while True:
    # Read the next frame from the video stream
    ret, frame = video_stream.read()
    if not ret:
        break

    # Detect objects in the frame using the object detection model
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    detections = model.forward()

    # Apply non-maximum suppression to remove duplicate detections
    boxes = []
    confidences = []
    class_ids = []
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(object_detection[0] * frame.shape[1])
                center_y = int(object_detection[1] * frame.shape[0])
                width = int(object_detection[2] * frame.shape[1])
                height = int(object_detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Update the tracker with the detected objects
    for i in indices:
        i = i[0]
        box = boxes[i]
        tracker.init(frame, tuple(box))

    # Update the tracker with new object detections and remove old tracks
    ok, boxes = tracker.update(frame)
    for i, box in enumerate(boxes):
        if ok:
            left, top, width, height = [int(v) for v in box]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # Display the output frame with object detections and tracks
    cv2.imshow('Real-Time Object Detection and Tracking System', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
video_stream.release()
cv2.destroyAllWindows()
