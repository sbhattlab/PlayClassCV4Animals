# Converting DLC dataset to YOLO
- Download base model for pose estimation (e.g. "yolo11x-pose.pt")
- Run `dlc2yolo.py` to convert DLC dataset to YOLO-compatible training dataset
    - Run `visualize_yolo_pose.py` and inspect conversion quality
- Run train with `from_dlc2yolo_converted_train.sh`
- Run prediction with `from_dlc2yolo_converted_predict.py`