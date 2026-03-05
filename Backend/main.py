from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import sys
import os

# Add the root 'TrustVision-AI' directory to the path so Backend can see ml/ and database/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference import DeepfakeDetector
from database.database import SessionLocal, engine
from database import models

# Create the database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Initialize the Deepfake ML Detector globally so it loads once on startup
detector = DeepfakeDetector()

app = FastAPI(
    title="TrustVision AI",
    description="Enterprise Media Authenticity Platform APIs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to TrustVision AI API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Dependency to get a Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/scan")
async def scan_media(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Receives an uploaded image, runs the deepfake ML model,
    saves the prediction to the database, and returns the result/heatmap.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported currently.")
        
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # 1. Run inference on the ML model
        result = detector.predict_image(image_bytes)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"ML Processing Error: {result.get('error')}")
            
        # 2. Save the result to our database
        scan_record = models.ScanResult(
            filename=file.filename,
            authenticity_score=result["authenticity_score"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            user_id=1 # Default user since auth isn't wired yet
        )
        db.add(scan_record)
        db.commit()
        db.refresh(scan_record)
        
        # 3. Return the response to the frontend
        return {
            "id": scan_record.id,
            "filename": scan_record.filename,
            "authenticity_score": scan_record.authenticity_score,
            "prediction": scan_record.prediction,
            "confidence": scan_record.confidence,
            "risk_level": scan_record.risk_level,
            "heatmap_base64": result["heatmap_base64"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
