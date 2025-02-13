from ultralytics import YOLO

# Load the trained model
model = YOLO("yolov8n.pt")


metrics = model.val(data='coco128.yaml')
