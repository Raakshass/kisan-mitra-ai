import os
from dotenv import load_dotenv
load_dotenv()
print("DEBUG: Loaded GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))

import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configure Gemini API key from .env
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI configured successfully!")
else:
    print("⚠️ GEMINI_API_KEY not found. Make sure to set it in a .env file.")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "🌱 Kisan Mitra AI Backend Running!",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": GEMINI_API_KEY is not None,
        "version": "2.0.0"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"success": False, "error": "Message cannot be empty"}), 400

        response_text = (
            get_gemini_chat_response(user_message)
            if GEMINI_API_KEY else
            get_fallback_response(user_message)
        )
        
        return jsonify({
            "success": True,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
            "powered_by": "Google Gemini AI" if GEMINI_API_KEY else "Local AI"
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to process message"
        }), 500

@app.route('/api/analyze-crop', methods=['POST'])
def analyze_crop():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        analysis_result = (
            analyze_crop_with_gemini(image_data)
            if GEMINI_API_KEY else
            get_fallback_analysis()
        )
        
        return jsonify({
            "success": True,
            **analysis_result
        })
        
    except Exception as e:
        print(f"Crop analysis error: {e}")
        return jsonify({
            "success": False,
            "error": "Analysis failed. Please try again."
        }), 500

# ✅ UPDATED FUNCTION — replies in the same language as question
def get_gemini_chat_response(user_message: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""You are an expert agricultural advisor for Indian farmers.

Farmer's Question: {user_message}

Provide helpful farming advice in the same language as the question. Reply only in the language of the question. Include:
1. Direct answer to the question
2. Practical steps with measurements
3. Preventive measures if applicable

Format with emojis and clear sections. Be friendly and practical.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini chat error: {e}")
        return get_fallback_response(user_message)

def analyze_crop_with_gemini(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """You are an expert plant pathologist and agricultural advisor. Analyze this crop image for:

1. Disease Identification: What disease or condition do you see?
2. Confidence Level: How confident are you (0.0 to 1.0)?
3. Treatment (Hindi): Treatment in Hindi for Indian farmers
4. Treatment (English): Treatment in English
5. Severity: healthy/mild/moderate/severe

Please provide a JSON response with these exact fields:
- disease: string
- confidence: float
- treatment_hindi: string
- treatment: string
- severity: string

Focus on practical, actionable advice for farmers."""

        response = model.generate_content([prompt, image])
        response_text = response.text
        print("🔍 Raw Gemini Response:\n", response_text)

        json_start = response_text.find('```json')
        if json_start != -1:
            json_start += len('```json')
            json_end = response_text.find('```', json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text.strip()

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            return parse_text_response(response_text)

        required_fields = ['disease', 'confidence', 'treatment_hindi', 'treatment', 'severity']
        for field in required_fields:
            if field not in result:
                result[field] = 'Not specified'

        try:
            result['confidence'] = float(result.get('confidence', 0.75))
        except:
            result['confidence'] = 0.75

        return result

    except Exception as e:
        print(f"Gemini vision analysis error: {e}")
        return get_fallback_analysis()

def parse_text_response(text):
    return {
        "disease": "AI Analysis Complete",
        "confidence": 0.85,
        "treatment_hindi": f"विश्लेषण: {text[:200]}...",
        "treatment": f"Analysis: {text[:200]}...",
        "severity": "moderate"
    }

def get_fallback_response(user_message):
    return f"""🌱 **किसान मित्र AI** (Local Mode)

आपका सवाल: "{user_message}"

**सामान्य सुझाव:**
- मिट्टी की जांच नियमित रूप से कराएं
- संतुलित खाद का उपयोग करें  
- पानी की उचित व्यवस्था करें
- फसल चक्र अपनाएं

📞 **तत्काल सहायता:** 1800-180-1551 (Kisan Call Center)

⚠️ *Gemini AI से जुड़ने के लिए API key सेट करें।*"""

def get_fallback_analysis():
    return {
        "disease": "स्थानीय विश्लेषण / Local Analysis",
        "confidence": 0.70,
        "treatment_hindi": "कृपया स्पष्ट तस्वीर अपलोड करें। सामान्य सुझाव: नीम तेल का छिड़काव करें।",
        "treatment": "Please upload a clear image. General advice: Apply neem oil spray.",
        "severity": "unknown"
    }

if __name__ == '__main__':
    print("=" * 50)
    print("🌱 KISAN MITRA AI BACKEND")
    print("=" * 50)
    print(f"🤖 Gemini AI: {'Enabled' if GEMINI_API_KEY else 'Disabled (set GEMINI_API_KEY in .env)'}")
    print(f"🌐 Starting on: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
