import os
from dotenv import load_dotenv 
import requests
load_dotenv()
key = os.getenv("GEMINI_API_KEY") 
print("key found:", key[:10] + "..." if key else "not found - .env wrong")