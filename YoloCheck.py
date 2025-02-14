from roboflow import Roboflow

rf = Roboflow(api_key="ThFg4vsVNX7z3J7mtySg")
project = rf.workspace().project("coco-128")
dataset = project.version(1).download("coco")


from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=20, imgsz=640)
