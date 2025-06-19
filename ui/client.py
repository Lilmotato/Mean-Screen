import requests
from config import API_URL


def analyze_text(text: str) -> dict:
    """Send user text to the API and return structured response."""
    payload = {"text": text}
    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"API Error: {response.status_code} - {response.text}")

    return response.json()
