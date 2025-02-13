from roboflow import Roboflow

rf = Roboflow(api_key="ThFg4vsVNX7z3J7mtySg")
project = rf.workspace().project("coco-128")  # Example dataset
dataset = project.version(1).download("coco")


from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano version (fast, lightweight)

# Train model on your dataset
model.train(data="data.yaml", epochs=20, imgsz=640)
