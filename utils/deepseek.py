import json
import requests
import streamlit as st

def query_deepseek(message, api_key, context=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    language = st.session_state.language
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Plant Disease Diagnosis"
    }

    system_prompt = {
        "uz": "Siz faqat o‘simlik kasalliklari va qishloq xo‘jaligi bo‘yicha maslahat beruvchi botsiz. Faqat qisqa va aniq javob ber.",
        "ru": "Вы спец только по болезням растений и сельскому хозяйству. Отвечай кратко и по делу.",
        "en": "You are a bot specialized only in plant diseases and agriculture. Answer briefly and clearly."
    }[language]

    if context:
        system_prompt += f"\nKontekst: {context}"

    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "max_tokens": 1000,
        "temperature": 0.5
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Xato: API bilan bog‘lanishda muammo: {str(e)}"

def format_bot_response(text):
    text = text.strip()
    if text.lower().startswith("uzum"):
        text = "### 🟢 " + text
    text = text.replace("1. ", "1. **").replace("2. ", "2. **").replace(": ", ":** ")
    return text
