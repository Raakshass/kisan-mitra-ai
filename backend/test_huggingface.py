import requests
import sys

# Your Hugging Face API key
API_KEY = "your-huggingface-api-key-here"

# Models to test
models_to_test = [
    "microsoft/DialoGPT-large",
    "facebook/blenderbot-3B",
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    # Add any other models you want to test
]

def test_model(model_name):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    payload = {
        "inputs": "Hello, I need farming advice",
        "parameters": {"max_new_tokens": 50}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✅ {model_name}: WORKING")
            return True
        else:
            print(f"❌ {model_name}: ERROR {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {model_name}: EXCEPTION {str(e)}")
        return False

if __name__ == "__main__":
    working_models = []
    
    for model in models_to_test:
        if test_model(model):
            working_models.append(model)
    
    print("\n=== WORKING MODELS ===")
    for model in working_models:
        print(f"- \"{model}\",")
