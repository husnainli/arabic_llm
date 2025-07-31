import requests
import os

# Requires a Hugging Face API token
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_ID = "inceptionai/jais-13b-chat"  # hosted on Hugging Face

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def query_jais(prompt, max_tokens=512, temperature=0.7):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Jais API Error: {response.status_code} — {response.text}")
    result = response.json()
    return result[0]["generated_text"]
