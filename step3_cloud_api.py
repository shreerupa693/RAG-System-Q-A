import os
import requests
from dotenv import load_dotenv
# Paste the API key you copied from Google AI Studio right between the quotes below:
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def ask_cloud_llm(query: str, retrieved_context: list[dict]) -> str:
    """
    Step 3 Practical Implementation: Offloads the heavy text reasoning to the cloud.
    Takes your question and the exact matching slides found by your local database.
    """
    # 1. Format the database text snippets cleanly for the cloud engine to read
    context_str = "\n\n".join([
        f"[Slide Source: {c['metadata']['source']}, Page: {c['metadata']['page']}]\nText: {c['text']}" 
        for c in retrieved_context
    ])
    
    # 2. Strict Prompt Injection Prevention & Context Anchoring
    prompt = (
        "You are an expert, strict NPTEL examiner for the course 'Introduction to Industry 4.0 and IIoT'.\n"
        "Your task is to answer the user's question or fulfill their quiz request using ONLY the context provided below.\n"
        "CRITICAL RULES:\n"
        "1. Do not use any outside knowledge. If the answer cannot be found in the context, state 'Information not found in the lecture slides.'\n"
        "2. Be technically precise. Always include exact IEEE standards, protocol names, values, or layers if they are in the text.\n"
        "3. Always cite the Source and Page number from the context at the end of your explanation.\n\n"
        f"CONTEXT MATERIAL:\n{context_str}\n\n"
        f"USER REQUEST / QUESTION: {query}\n\n"
        "ANSWER:"
    )

    # 3. Formulate the Web Endpoint and Payload according to the Gemini API protocol
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    # 4. Ship the data over the network via an HTTP POST request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"[NETWORK ERROR] Cloud API returned Status Code {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "[TIMEOUT ERROR] The cloud server took too long to reply. Check your internet connection."
    except Exception as e:
        return f"[SYSTEM ERROR] Failed to complete network handshake: {str(e)}"
