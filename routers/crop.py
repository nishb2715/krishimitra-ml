from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.crop_predictor import crop_predictor

router = APIRouter(prefix="/crop", tags=["Crop Doctor"])

@router.post("/diagnose")
async def diagnose_crop(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large, max 10MB")
    
    result = crop_predictor.predict(image_bytes)
    return result

@router.get("/classes")
async def get_classes():
    return {
        "total_classes": len(crop_predictor.labels),
        "classes": [
            {"id": k, "name": v["class_name"], "odia": v["odia_name"]}
            for k, v in crop_predictor.labels.items()
        ]
    }