# ============================================================
# CROP DISEASE DETECTION API - PRODUCTION READY
# ============================================================

import os
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import timm
import io
import logging
from datetime import datetime

# ── LOGGING ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CropDiseaseAPI")

# ── APP SETUP ──────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Detection API",
    description="Tamil Nadu Agriculture - AI Disease Detector",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONSTANTS ──────────────────────────────────────────────
MIN_CONFIDENCE = 60.0
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}
device = torch.device("cpu")
MODEL_PATH = "model.pth"

# ── LOAD MODEL ─────────────────────────────────────────────
logger.info("Loading AI model...")
model = None
CLASS_NAMES = []
NUM_CLASSES = 0

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        CLASS_NAMES = checkpoint.get('class_names', [])
        NUM_CLASSES = len(CLASS_NAMES)
        ARCH = checkpoint.get('architecture', 'efficientnet_b3')

        model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        logger.info(f"✅ Model loaded: {ARCH}, {NUM_CLASSES} classes")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
else:
    logger.warning(f"⚠️ Model file not found at {MODEL_PATH}. Running in demo mode.")

# ── IMAGE TRANSFORM ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── API ROUTES ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "active",
        "api": "Crop Disease Detection API",
        "version": "1.0.0",
        "classes": NUM_CLASSES,
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": NUM_CLASSES,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded. Please try again later.")
    
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"File must be jpg/png/webp. Got: {file.content_type}")
    
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Max 10MB.")
    
    if len(contents) < 1000:
        raise HTTPException(400, "File too small. Upload a clear crop leaf photo.")
    
    try:
        # Decode image
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(400, "Corrupted or invalid image file.")
        
        w, h = img.size
        if w < 100 or h < 100:
            raise HTTPException(400, f"Image too small ({w}x{h}). Please use a clearer photo.")
        
        # Predict
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        confidence, pred_idx = probs.max(0)
        conf_pct = round(confidence.item() * 100, 2)
        class_name = CLASS_NAMES[pred_idx.item()] if CLASS_NAMES else "unknown"
        
        is_diseased = 'healthy' not in class_name.lower()
        
        return JSONResponse(content={
            "status": "success",
            "is_diseased": is_diseased,
            "is_diseased_ta": "ஆம், நோய் கண்டறியப்பட்டது." if is_diseased else "இல்லை, பயிர் ஆரோக்கியமாக உள்ளது.",
            "disease_class": class_name,
            "confidence_pct": conf_pct,
            "message": "Analysis complete"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
