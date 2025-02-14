from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict("down.jpg")

results[0].show()
