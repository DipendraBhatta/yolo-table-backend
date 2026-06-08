import io
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# 1. Initialize FastAPI Application
app = FastAPI(
    title="YOLOv8 Table Layout Detector API",
    description="API to extract tables, row headers, and column headers from document images.",
    version="1.0.0"
)
allow_origins=[
    "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://192.168.1.84:3000",
        "http://192.168.1.84:3001"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target the weights path relative to this file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "best.pt"

model = None


if MODEL_PATH.exists():
    print(f" Loading trained weights from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))
else:
    print(f" Warning: Model weights not found at {MODEL_PATH}. Prediction endpoint will be disabled.")


# 2. Health Check Endpoint

@app.get("/")
async def root():
    return {"message": "Service is running and ready to accept requests."}


# Status endpoint — model info
@app.get("/status")
def get_status():
    status_str = "best.pt loaded" if model is not None else "No model loaded (offline)"
    return {
        "message": "YOLO FastAPI is running", 
        "model": status_str
    }


@app.post("/predict")
async def predict_table_layout(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through the YOLO model,
    and returns detected bounding boxes, confidence scores, and class labels.
    """

    # Safety Shield: Block execution if model is not loaded or has been deleted
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI Model weights are missing or unloaded on this server. Prediction is currently disabled."
        )

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
    

# 4. Model Lifecycle Management Endpoints

@app.put("/update_model")
async def update_model(weight_path: str):
    """
    Dynamically allows manual swapping or reloading of weights via path parameter.
    """
    try:
        global model
        target_path = Path(weight_path)
        if not target_path.is_absolute():
            target_path = BASE_DIR / weight_path

        if not target_path.exists():
            raise FileNotFoundError(f"No file found at {target_path}")

        model = YOLO(str(target_path))  # reload with new weights
        return {"message": f"Model updated perfectly to {target_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@app.delete("/delete_model")
async def delete_model():
    """
    Safely unloads the active YOLO model from RAM memory and disables the prediction endpoint.
    """
    global model
    if model is None:
        return {"message": "No active model was loaded to delete."}
    
    try:
        # Clear the model instance from system memory
        del model
        model = None  # Reset variable back to blank state
        return {"message": "Model successfully unloaded and removed from memory."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear model from memory: {str(e)}")