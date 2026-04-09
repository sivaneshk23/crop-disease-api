# ============================================================
# app.py - COMPLETE PRODUCTION API FOR CROP DISEASE DETECTION
# Tamil + English output for your Supabase chatbot
# Works with your Phase 1 model (95-96% accuracy)
# ============================================================

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
import json
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
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    CLASS_NAMES = checkpoint['class_names']
    NUM_CLASSES = len(CLASS_NAMES)
    ARCH = checkpoint.get('architecture', 'efficientnet_b3')

    model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logger.info(f"✅ Model loaded: {ARCH}, {NUM_CLASSES} classes")
except Exception as e:
    logger.error(f"❌ Model load FAILED: {e}")
    raise RuntimeError(f"Model load failed: {e}")

# ── IMAGE TRANSFORM ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── COMPLETE DISEASE KNOWLEDGE BASE (TAMIL + ENGLISH) ──────
DISEASE_KB = {
    # RICE
    "rice_bacterial_blight": {
        "en_name": "Rice Bacterial Blight",
        "ta_name": "நெல் பாக்டீரியல் ப்லைட் நோய்",
        "medicine": "Copper Oxychloride 50% WP",
        "dose_en": "3g per litre",
        "dose_ta": "1 லிட்டருக்கு 3 கிராம்",
        "price": "₹180–₹250 per kg",
        "shop_en": "TNAU Krishi Vigyan Kendra",
        "shop_ta": "TNAU கிருஷி விஞ்ஞான் கேந்திரா",
        "frequency_en": "Every 10 days, 3 times",
        "frequency_ta": "10 நாட்களுக்கு ஒரு முறை, 3 தடவை",
        "yield_loss": "Up to 70%"
    },
    "rice_blast": {
        "en_name": "Rice Blast",
        "ta_name": "நெல் ப்லாஸ்ட் நோய்",
        "medicine": "Tricyclazole 75% WP",
        "dose_en": "6g per 10 litres",
        "dose_ta": "10 லிட்டருக்கு 6 கிராம்",
        "price": "₹350–₹500 per 100g",
        "shop_en": "Certified agri dealers",
        "shop_ta": "அங்கீகரிக்கப்பட்ட வேளாண் கடைகள்",
        "frequency_en": "At tillering and panicle stage",
        "frequency_ta": "நாற்று மற்றும் கதிர் நிலையில்",
        "yield_loss": "Up to 80%"
    },
    "rice_brown_spot": {
        "en_name": "Rice Brown Spot",
        "ta_name": "நெல் பழுப்பு புள்ளி நோய்",
        "medicine": "Mancozeb 75% WP",
        "dose_en": "2.5g per litre",
        "dose_ta": "1 லிட்டருக்கு 2.5 கிராம்",
        "price": "₹150–₹200 per kg",
        "shop_en": "All agri shops",
        "shop_ta": "அனைத்து வேளாண் கடைகளிலும்",
        "frequency_en": "Every 10-14 days, 2-3 times",
        "frequency_ta": "10-14 நாட்களுக்கு ஒரு முறை, 2-3 தடவை",
        "yield_loss": "Up to 45%"
    },
    "rice_healthy": {
        "en_name": "Healthy Rice",
        "ta_name": "ஆரோக்கியமான நெல் பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # TOMATO
    "tomato_early_blight": {
        "en_name": "Tomato Early Blight",
        "ta_name": "தக்காளி ஆரம்ப கருகல் நோய்",
        "medicine": "Mancozeb 75% WP",
        "dose_en": "2g per litre",
        "dose_ta": "1 லிட்டருக்கு 2 கிராம்",
        "price": "₹150–₹200 per kg",
        "shop_en": "All agri shops",
        "shop_ta": "அனைத்து வேளாண் கடைகளிலும்",
        "frequency_en": "Every 7 days for 3 weeks",
        "frequency_ta": "3 வாரங்களுக்கு 7 நாட்களுக்கு ஒரு முறை",
        "yield_loss": "Up to 50%"
    },
    "tomato_late_blight": {
        "en_name": "Tomato Late Blight",
        "ta_name": "தக்காளி தாமத கருகல் நோய்",
        "medicine": "Metalaxyl 8% + Mancozeb 64% WP",
        "dose_en": "2.5g per litre",
        "dose_ta": "1 லிட்டருக்கு 2.5 கிராம்",
        "price": "₹300–₹450 per kg",
        "shop_en": "Certified agri shops",
        "shop_ta": "அங்கீகரிக்கப்பட்ட வேளாண் கடைகள்",
        "frequency_en": "Every 7 days until controlled",
        "frequency_ta": "கட்டுப்பாட்டிற்கு வரும் வரை 7 நாட்களுக்கு ஒரு முறை",
        "yield_loss": "Up to 100%"
    },
    "tomato_bacterial_spot": {
        "en_name": "Tomato Bacterial Spot",
        "ta_name": "தக்காளி பாக்டீரியல் புள்ளி நோய்",
        "medicine": "Copper Hydroxide 77% WP",
        "dose_en": "2g per litre",
        "dose_ta": "1 லிட்டருக்கு 2 கிராம்",
        "price": "₹250–₹350 per kg",
        "shop_en": "Agri shops and BigHaat",
        "shop_ta": "வேளாண் கடைகள் மற்றும் BigHaat",
        "frequency_en": "Every 7 days, 3 times",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை, 3 தடவை",
        "yield_loss": "Up to 40%"
    },
    "tomato_healthy": {
        "en_name": "Healthy Tomato",
        "ta_name": "ஆரோக்கியமான தக்காளி பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # BANANA
    "banana_black_sigatoka": {
        "en_name": "Banana Black Sigatoka",
        "ta_name": "வாழை கருப்பு சிகடோகா நோய்",
        "medicine": "Propiconazole 25% EC",
        "dose_en": "1ml per litre",
        "dose_ta": "1 லிட்டருக்கு 1 மில்லி",
        "price": "₹500–₹700 per 500ml",
        "shop_en": "Agri shops and AgroStar",
        "shop_ta": "வேளாண் கடைகள் மற்றும் AgroStar",
        "frequency_en": "Every 21 days",
        "frequency_ta": "21 நாட்களுக்கு ஒரு முறை",
        "yield_loss": "Up to 50%"
    },
    "banana_healthy": {
        "en_name": "Healthy Banana",
        "ta_name": "ஆரோக்கியமான வாழை பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # SUGARCANE
    "sugarcane_red_rot": {
        "en_name": "Sugarcane Red Rot",
        "ta_name": "கரும்பு சிவப்பு அழுகல் நோய்",
        "medicine": "Carbendazim 50% WP",
        "dose_en": "1g per litre - drench soil",
        "dose_ta": "1 லிட்டருக்கு 1 கிராம் - மண்ணில் ஊற்றவும்",
        "price": "₹200–₹350 per kg",
        "shop_en": "District agri shops",
        "shop_ta": "மாவட்ட வேளாண் கடைகள்",
        "frequency_en": "Monthly, 3 times after removing infected stalks",
        "frequency_ta": "நோய் தாக்கிய கரும்பை நீக்கிய பின் மாதம் ஒரு முறை, 3 தடவை",
        "yield_loss": "Up to 60%"
    },
    "sugarcane_healthy": {
        "en_name": "Healthy Sugarcane",
        "ta_name": "ஆரோக்கியமான கரும்பு பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # POTATO
    "potato_early_blight": {
        "en_name": "Potato Early Blight",
        "ta_name": "உருளைக்கிழங்கு ஆரம்ப கருகல் நோய்",
        "medicine": "Mancozeb 75% WP",
        "dose_en": "2.5g per litre",
        "dose_ta": "1 லிட்டருக்கு 2.5 கிராம்",
        "price": "₹150–₹200 per kg",
        "shop_en": "All agri shops",
        "shop_ta": "அனைத்து வேளாண் கடைகளிலும்",
        "frequency_en": "Every 7 days, 3 times",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை, 3 தடவை",
        "yield_loss": "Up to 50%"
    },
    "potato_healthy": {
        "en_name": "Healthy Potato",
        "ta_name": "ஆரோக்கியமான உருளைக்கிழங்கு",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # CORN
    "corn_common_rust": {
        "en_name": "Corn Common Rust",
        "ta_name": "சோளம் பொதுவான துரு நோய்",
        "medicine": "Propiconazole 25% EC",
        "dose_en": "1ml per litre",
        "dose_ta": "1 லிட்டருக்கு 1 மில்லி",
        "price": "₹500–₹700 per 500ml",
        "shop_en": "Agri shops",
        "shop_ta": "வேளாண் கடைகள்",
        "frequency_en": "Every 14 days, 2 times",
        "frequency_ta": "14 நாட்களுக்கு ஒரு முறை, 2 தடவை",
        "yield_loss": "Up to 40%"
    },
    "corn_healthy": {
        "en_name": "Healthy Corn",
        "ta_name": "ஆரோக்கியமான சோளம் பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # APPLE
    "apple_scab": {
        "en_name": "Apple Scab",
        "ta_name": "ஆப்பிள் ஸ்காப் நோய்",
        "medicine": "Mancozeb 75% WP",
        "dose_en": "2.5g per litre",
        "dose_ta": "1 லிட்டருக்கு 2.5 கிராம்",
        "price": "₹150–₹200 per kg",
        "shop_en": "Agri shops",
        "shop_ta": "வேளாண் கடைகள்",
        "frequency_en": "Every 7-10 days during wet season",
        "frequency_ta": "ஈரமான பருவத்தில் 7-10 நாட்களுக்கு ஒரு முறை",
        "yield_loss": "Up to 70%"
    },
    "apple_healthy": {
        "en_name": "Healthy Apple",
        "ta_name": "ஆரோக்கியமான ஆப்பிள் பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # GRAPE
    "grape_black_rot": {
        "en_name": "Grape Black Rot",
        "ta_name": "திராட்சை கருப்பு அழுகல் நோய்",
        "medicine": "Mancozeb 75% WP",
        "dose_en": "2.5g per litre",
        "dose_ta": "1 லிட்டருக்கு 2.5 கிராம்",
        "price": "₹150–₹200 per kg",
        "shop_en": "All agri shops",
        "shop_ta": "அனைத்து வேளாண் கடைகளிலும்",
        "frequency_en": "Every 7-10 days",
        "frequency_ta": "7-10 நாட்களுக்கு ஒரு முறை",
        "yield_loss": "Up to 80%"
    },
    "grape_healthy": {
        "en_name": "Healthy Grape",
        "ta_name": "ஆரோக்கியமான திராட்சை பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
    
    # PEPPER
    "pepper_bacterial_spot": {
        "en_name": "Pepper Bacterial Spot",
        "ta_name": "மிளகாய் பாக்டீரியல் புள்ளி நோய்",
        "medicine": "Copper Hydroxide 77% WP",
        "dose_en": "2g per litre",
        "dose_ta": "1 லிட்டருக்கு 2 கிராம்",
        "price": "₹250–₹350 per kg",
        "shop_en": "All agri shops",
        "shop_ta": "அனைத்து வேளாண் கடைகளிலும்",
        "frequency_en": "Every 7 days, 3 times",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை, 3 தடவை",
        "yield_loss": "Up to 40%"
    },
    "pepper_healthy": {
        "en_name": "Healthy Pepper",
        "ta_name": "ஆரோக்கியமான மிளகாய் பயிர்",
        "medicine": "No medicine needed",
        "dose_en": "No treatment",
        "dose_ta": "சிகிச்சை தேவையில்லை",
        "price": "N/A",
        "shop_en": "No medicine needed",
        "shop_ta": "மருந்து தேவையில்லை",
        "frequency_en": "Monitor every 7 days",
        "frequency_ta": "7 நாட்களுக்கு ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None"
    },
}

def get_disease_info(class_name):
    """Get disease info with smart fallback"""
    if class_name in DISEASE_KB:
        return DISEASE_KB[class_name]
    
    is_healthy = 'healthy' in class_name.lower()
    crop = class_name.split('_')[0].capitalize()
    
    return {
        "en_name": f"Healthy {crop}" if is_healthy else f"{crop} Disease",
        "ta_name": f"ஆரோக்கியமான {crop}" if is_healthy else f"{crop} நோய்",
        "medicine": "No medicine needed" if is_healthy else "Consult local agriculture officer",
        "dose_en": "No treatment" if is_healthy else "Contact agri officer",
        "dose_ta": "சிகிச்சை தேவையில்லை" if is_healthy else "வேளாண்மை அதிகாரியை அணுகவும்",
        "price": "N/A" if is_healthy else "Contact local shop",
        "shop_en": "No medicine needed" if is_healthy else "Visit nearest agri shop",
        "shop_ta": "மருந்து தேவையில்லை" if is_healthy else "அருகிலுள்ள வேளாண் கடையை அணுகவும்",
        "frequency_en": "Monitor weekly",
        "frequency_ta": "வாரம் ஒரு முறை கண்காணிக்கவும்",
        "yield_loss": "None" if is_healthy else "Unknown"
    }

def get_affected_pct(confidence, is_diseased):
    if not is_diseased:
        return "0%"
    if confidence >= 92: return "75–100%"
    elif confidence >= 80: return "50–75%"
    elif confidence >= 65: return "25–50%"
    else: return "10–25%"

# ── API ROUTES ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "active",
        "api": "Crop Disease Detection API",
        "version": "1.0.0",
        "classes": NUM_CLASSES,
        "description": "Tamil Nadu Agriculture AI"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": NUM_CLASSES,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES, "total": NUM_CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Send crop leaf photo → get Tamil + English disease info.
    """
    request_id = datetime.now().strftime("%H%M%S%f")
    logger.info(f"[{request_id}] File: {file.filename} | Type: {file.content_type}")
    
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        logger.warning(f"[{request_id}] Bad type: {file.content_type}")
        raise HTTPException(400, f"File must be jpg/png/webp. Got: {file.content_type}")
    
    contents = await file.read()
    
    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        logger.warning(f"[{request_id}] Too large: {len(contents)} bytes")
        raise HTTPException(400, "File too large. Max 10MB.")
    
    if len(contents) < 1000:
        logger.warning(f"[{request_id}] Too small: {len(contents)} bytes")
        raise HTTPException(400, "File too small. Upload a clear crop leaf photo.")
    
    try:
        # Decode image
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            logger.warning(f"[{request_id}] Corrupted image")
            raise HTTPException(400, "Corrupted or invalid image file.")
        
        # Check image size
        w, h = img.size
        if w < 100 or h < 100:
            logger.warning(f"[{request_id}] Too small: {w}x{h}")
            raise HTTPException(400, f"Image too small ({w}x{h}). Please use a clearer photo.")
        
        # Predict
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        confidence, pred_idx = probs.max(0)
        conf_pct = round(confidence.item() * 100, 2)
        class_name = CLASS_NAMES[pred_idx.item()]
        
        logger.info(f"[{request_id}] Prediction: {class_name} | Confidence: {conf_pct}%")
        
        # Low confidence check
        if conf_pct < MIN_CONFIDENCE:
            logger.warning(f"[{request_id}] Low confidence: {conf_pct}%")
            return JSONResponse(content={
                "status": "uncertain",
                "confidence_pct": conf_pct,
                "is_diseased": False,
                "is_diseased_en": f"Image not clear enough ({conf_pct:.0f}%). Please retake.",
                "is_diseased_ta": f"படம் தெளிவாக இல்லை ({conf_pct:.0f}%). மீண்டும் எடுக்கவும்.",
                "suggestion_en": "Ensure leaf fills frame, good lighting, no blur",
                "suggestion_ta": "இலை முழுவதும் தெரியும்படி, நல்ல வெளிச்சத்தில், தெளிவாக எடுக்கவும்"
            })
        
        # Build response
        is_diseased = 'healthy' not in class_name.lower()
        info = get_disease_info(class_name)
        affected = get_affected_pct(conf_pct, is_diseased)
        
        # Top 3 predictions
        top3_v, top3_i = torch.topk(probs, k=min(3, NUM_CLASSES))
        top3 = [
            {"class": CLASS_NAMES[i.item()], "confidence": round(v.item() * 100, 2)}
            for v, i in zip(top3_v, top3_i)
        ]
        
        response = {
            "status": "success",
            
            # Q1 - Is diseased?
            "is_diseased": is_diseased,
            "is_diseased_en": "Yes, disease detected." if is_diseased else "No, crop is healthy.",
            "is_diseased_ta": "ஆம், நோய் கண்டறியப்பட்டது." if is_diseased else "இல்லை, பயிர் ஆரோக்கியமாக உள்ளது.",
            
            # Q2 - Disease info
            "disease_class": class_name,
            "disease_name_en": info["en_name"],
            "disease_name_ta": info["ta_name"],
            "confidence_pct": conf_pct,
            "affected_percentage": affected,
            "yield_loss": info["yield_loss"],
            
            # Q3 - Solution + price + shop
            "medicine": info["medicine"],
            "dose_en": info["dose_en"],
            "dose_ta": info["dose_ta"],
            "frequency_en": info["frequency_en"],
            "frequency_ta": info["frequency_ta"],
            "price_per_unit": info["price"],
            "shop_en": info["shop_en"],
            "shop_ta": info["shop_ta"],
            
            "top3": top3,
            "model_version": "EfficientNet-B3-v1"
        }
        
        logger.info(f"[{request_id}] Response sent: {class_name}, {conf_pct}%")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Analysis failed. Please try again."}
        )