from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano version (fast, lightweight)

# Train model on your dataset
model.train(data="coco128.yaml", epochs=20, imgsz=640)
