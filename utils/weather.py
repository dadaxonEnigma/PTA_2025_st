import requests
import streamlit as st

def get_weather(lat, lon):
    api_key = st.secrets.get("OPENWEATHER_API")
    if not api_key:
        return None
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url).json()
        temp = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        return temp, humidity
    except Exception as e:
        print("❌ Ошибка при получении погоды:", e)
        return None