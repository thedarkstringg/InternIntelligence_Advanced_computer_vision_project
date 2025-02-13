from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

api = FastAPI()

#Loading my YoloV8 Train Model
model = YOLO('yolov8n.pt')

@api.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read()))
    
    #performing inference
    results = model(image)

    results[0].show() 
    
    #extracting predictions
    predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                "class": model.names[int(box.cls)],
                "confidence": round(box.conf.item(), 2),
                "x_min": int(box.xyxy[0][0]),
                "y_min": int(box.xyxy[0][1]),
                "x_max": int(box.xyxy[0][2]),
                "y_max": int(box.xyxy[0][3])
            })
    
    return JSONResponse(content={"detections": predictions})

