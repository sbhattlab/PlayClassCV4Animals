from ultralytics import YOLO

model = YOLO("YOLO/runs/pose/train/weights/best.pt")

results = model.predict(
    source="data/test.mp4",
    save=True,  # writes an annotated video
    save_txt=False,
    conf=0.5,
    stream=False,
)
