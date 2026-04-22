from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session
import sys
import os

# Add the root 'TrustVision-AI' directory to the path so Backend can see ml/ and database/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference import DeepfakeDetector
from Backend.reporting import MODEL_VERSION, build_explanation_summary, generate_pdf_report
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
        
        # Reverse the prediction (REAL -> FAKE, FAKE -> REAL)
        reversed_prediction = "FAKE" if result["prediction"] == "REAL" else "REAL"
        reversed_authenticity_score = 100 - result["authenticity_score"]
            
        # 2. Save the result to our database
        scan_record = models.ScanResult(
            filename=file.filename,
            authenticity_score=reversed_authenticity_score,
            prediction=reversed_prediction,
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
            "timestamp": scan_record.timestamp.isoformat() if scan_record.timestamp else None,
            "model_version": MODEL_VERSION,
            "report_url": f"/reports/{scan_record.id}",
            "heatmap_base64": result["heatmap_base64"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{scan_id}")
def download_report(scan_id: int, db: Session = Depends(get_db)):
    scan_record = db.query(models.ScanResult).filter(models.ScanResult.id == scan_id).first()
    if scan_record is None:
        raise HTTPException(status_code=404, detail="Scan result not found")

    try:
        pdf_bytes, report_filename, _ = generate_pdf_report(scan_record)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{report_filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")
