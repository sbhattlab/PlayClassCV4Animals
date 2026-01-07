import cv2
from ultralytics import YOLO

# Load a small, fast model (yolov8n). You can switch to yolov8s/m/l/x for more accuracy.
model = YOLO("yolov8n.pt")

# Choose your source: 0 = default webcam; or use a file path for video, e.g., "video.mp4"
cap = cv2.VideoCapture(0)

# If you have a GPU, you can set device=0 in the call for speed. Example:
# results = model(frame, device=0, conf=0.25, iou=0.45)


def draw_label(img, label, x1, y1):
    """Draw a filled rectangle behind the text for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(
        img, (x1, y1 - h - 6), (x1 + w + 4, y1), (0, 0, 0), -1
    )  # black background
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Run inference on the frame
    results = model(frame, conf=0.25, iou=0.45)  # adjust thresholds as needed
    r = results[0]  # batch of 1

    # Loop over detections
    for box in r.boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        # Class and confidence
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Draw bounding box
        color = (0, 255, 0)  # green (you can pick per-class colors if you like)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1)

    cv2.imshow("Object detection (YOLOv8)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
