from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import crop, livestock

app = FastAPI(
    title="KrishiMitra ML API",
    description="Crop disease detection and livestock health monitoring for Odisha farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crop.router)
app.include_router(livestock.router)

@app.get("/")
async def root():
    return {
        "app": "KrishiMitra ML API",
        "version": "1.0.0",
        "endpoints": {
            "crop_diagnose":      "POST /crop/diagnose",
            "livestock_diagnose": "POST /livestock/diagnose",
            "docs":               "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}