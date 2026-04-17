import os
import json
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import timm
import io
from datetime import datetime

# Configuration
MODEL_PATH = "model.pth"
MIN_CONFIDENCE = 60.0
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}

# Initialize app FIRST
app = FastAPI(title="Crop Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route - MUST be before model loading
@app.get("/")
def root():
    return {"status": "active", "api": "Crop Disease Detection API"}

@app.get("/health")
def health():
    model_loaded = os.path.exists(MODEL_PATH)
    return {"status": "healthy", "model_loaded": model_loaded}

# Load model
print("Loading model...")
CLASS_NAMES = []
NUM_CLASSES = 0
model = None

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        CLASS_NAMES = checkpoint.get('class_names', [])
        NUM_CLASSES = len(CLASS_NAMES)
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print(f"✅ Model loaded: {NUM_CLASSES} classes")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Disease Knowledge Base
DISEASE_KB = {
    "rice_bacterial_blight": {
        "en_name": "Rice Bacterial Blight", "ta_name": "நெல் பாக்டீரியல் ப்லைட் நோய்",
        "medicine": "Copper Oxychloride", "dose_ta": "1 லிட்டருக்கு 3 கிராம்",
        "price": "₹180–₹250 per kg", "shop_ta": "TNAU கிருஷி விஞ்ஞான் கேந்திரா"
    },
    "rice_blast": {
        "en_name": "Rice Blast", "ta_name": "நெல் ப்லாஸ்ட் நோய்",
        "medicine": "Tricyclazole", "dose_ta": "10 லிட்டருக்கு 6 கிராம்",
        "price": "₹350–₹500 per 100g", "shop_ta": "அங்கீகரிக்கப்பட்ட வேளாண் கடைகள்"
    },
    "rice_healthy": {
        "en_name": "Healthy Rice", "ta_name": "ஆரோக்கியமான நெல்",
        "medicine": "No medicine", "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A", "shop_ta": "மருந்து தேவையில்லை"
    }
}

def get_disease_info(class_name):
    if class_name in DISEASE_KB:
        return DISEASE_KB[class_name]
    return {
        "en_name": class_name, "ta_name": class_name,
        "medicine": "Consult officer", "dose_ta": "வேளாண்மை அதிகாரியை அணுகவும்",
        "price": "Contact shop", "shop_ta": "அருகிலுள்ள வேளாண் கடை"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "File must be jpg/png/webp")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Max 10MB.")
    
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file")
    
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    confidence, pred_idx = probs.max(0)
    conf_pct = round(confidence.item() * 100, 2)
    class_name = CLASS_NAMES[pred_idx.item()] if CLASS_NAMES else "unknown"
    
    is_diseased = 'healthy' not in class_name.lower()
    info = get_disease_info(class_name)
    
    return JSONResponse(content={
        "status": "success",
        "is_diseased": is_diseased,
        "is_diseased_ta": "ஆம், நோய் கண்டறியப்பட்டது." if is_diseased else "இல்லை, பயிர் ஆரோக்கியமாக உள்ளது.",
        "disease_class": class_name,
        "disease_name_ta": info["ta_name"],
        "confidence_pct": conf_pct,
        "medicine": info["medicine"],
        "dose_ta": info["dose_ta"],
        "price_per_unit": info["price"],
        "shop_ta": info["shop_ta"]
    })
