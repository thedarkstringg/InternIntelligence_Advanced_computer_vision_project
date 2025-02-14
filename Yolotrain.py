from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(data="coco128.yaml", epochs=20, imgsz=640)
