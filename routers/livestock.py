from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.livestock_predictor import livestock_predictor

router = APIRouter(prefix="/livestock", tags=["Livestock Monitor"])

@router.post("/diagnose")
async def diagnose_livestock(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large, max 10MB")
    
    result = livestock_predictor.predict(image_bytes)
    return result

@router.get("/classes")
async def get_classes():
    return {
        "total_classes": len(livestock_predictor.labels),
        "classes": [
            {"id": k, "name": v["class_name"], "odia": v["odia_name"]}
            for k, v in livestock_predictor.labels.items()
        ]
    }