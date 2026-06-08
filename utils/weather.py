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
        city = response.get("name", "")
        country = response.get("sys", {}).get("country", "")
        location_name = f"{city}, {country}" if city and country else (city or country)
        return temp, humidity, location_name
    except Exception as e:
        print("❌ Ошибка при получении погоды:", e)
        return None


def get_forecast(lat, lon):
    api_key = st.secrets.get("OPENWEATHER_API")
    if not api_key:
        return None
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url).json()
        days = {}
        for entry in response["list"]:
            date = entry["dt_txt"].split(" ")[0]
            days.setdefault(date, {"temps": [], "humidities": []})
            days[date]["temps"].append(entry["main"]["temp"])
            days[date]["humidities"].append(entry["main"]["humidity"])

        forecast = [
            {
                "date": date,
                "temp": round(sum(values["temps"]) / len(values["temps"]), 1),
                "humidity": round(sum(values["humidities"]) / len(values["humidities"]))
            }
            for date, values in days.items()
        ]
        return forecast
    except Exception as e:
        print("❌ Ошибка при получении прогноза погоды:", e)
        return None