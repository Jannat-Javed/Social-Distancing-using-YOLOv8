import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist

# Constants
MIN_CONF = 0.3
MIN_DISTANCE = 50
VIDEO_PATH = 'pedestrians.mp4'
OUTPUT_PATH = 'output.avi'
DISPLAY = True

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = None
(W, H) = (None, None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (720, int(frame.shape[0] * 720 / frame.shape[1])))
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Detect people using YOLOv8
    results = model.predict(frame, conf=MIN_CONF, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() else []

    centroids, boxes = [], []
    for *box, conf, cls in detections:
        if int(cls) == 0:  # 0 = 'person' in COCO
            x1, y1, x2, y2 = map(int, box)
            cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cX, cY))
            boxes.append((x1, y1, x2, y2))

    violate = set()
    if len(centroids) >= 2:
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    # Draw boxes and violations
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        color = (0, 0, 255) if i in violate else (0, 255, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, centroids[i], 5, color, -1)

    # Put violation count
    text = f"Social Distancing Violations: {len(violate)}"
    cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save output
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H), True)

    writer.write(frame)

    if DISPLAY:
        cv2.imshow("Social Distancing Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

# Cleanup
cap.release()
writer.release()
cv2.destroyAllWindows()
