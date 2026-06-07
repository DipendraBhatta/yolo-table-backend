import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# 1. Initialize FastAPI Application
app = FastAPI(
    title="YOLOv8 Table Layout Detector API",
    description="API to extract tables, row headers, and column headers from document images.",
    version="1.0.0"
)

# Target the weights path relative to this file location
MODEL_PATH = "weights/best.pt"

if os.path.exists(MODEL_PATH):
    print(f" Loading trained weights from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
else:
    raise RuntimeError(f" Critical Error: Model weights not found at {MODEL_PATH}")


# 2. Health Check Endpoint

@app.get("/")
async def root():
    return {"message": "Service is up"}

# Status endpoint — model info
@app.get("/status")
def get_status():
    return {"message": "YOLO FastAPI is running", "model": "best.pt"}
@app.post("/predict")
async def predict_table_layout(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through the YOLO model,
    and returns detected bounding boxes, confidence scores, and class labels.
    """
    # Validate file type extension
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a JPG or PNG.")

    try:
        # Read the raw uploaded file bytes directly into memory
        image_bytes = await file.read()
        
        # Convert bytes to a PIL Image instance
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference matching your 800px high-resolution training setup
        results = model.predict(source=image, imgsz=800, conf=0.5, iou=0.5)
        
        # Parse output from the first image in the results list
        result = results[0]
        
        detections = []
        # Loop through found boxes and compile structural meta JSON dictionaries
        for box in result.boxes:
            # Extract box coordinates in xyxy tracking format
            xyxy = box.xyxy[0].tolist() 
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                }
            })

        return JSONResponse(content={
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
            "detections_count": len(detections),
            "detections": detections
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")
    

@app.put("/update_model")
async def update_model(weight_path: str):
        try:
            global model
            model = YOLO(weight_path)   # reload with new weights
            return {"message": f"Model updated to {weight_path}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")