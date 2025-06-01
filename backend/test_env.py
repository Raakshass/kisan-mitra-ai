import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.getcwd(), ".env")

print("Looking for .env at:", dotenv_path)
print("Exists:", os.path.exists(dotenv_path))

load_dotenv(dotenv_path=dotenv_path)

print("GEMINI_API_KEY =", os.getenv("GEMINI_API_KEY"))
