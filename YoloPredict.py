from ultralytics import YOLO

# Load the trained model
model = YOLO("yolov8n.pt")

# Test on a single image
results = model.predict("down.jpg")

# Show the results (visualize the predictions)
results[0].show()  # This will display the image with bounding boxes
