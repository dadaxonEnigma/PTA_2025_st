import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",  # Ваш ключ
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Plant Disease Diagnosis"
    },
    data=json.dumps({
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {"role": "user", "content": "Как лечить болезни растений?"}
        ]
    })
)
print(response.status_code)
print(response.text)