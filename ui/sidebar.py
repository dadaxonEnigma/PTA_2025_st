import streamlit as st
from data.translations import sidebar_content
from utils.weather import geocode_city

def render_sidebar(get_text, get_weather, treatment):
    st.header(get_text("sidebar_tools"))
    
    # Выбор языка
    lang_options = {
        "uz": "🇺🇿 O‘zbekcha",
        "en": "🇬🇧 English",
        "ru": "🇷🇺 Русский"
    }
    st.session_state.language = st.selectbox(
        "🌐 Tilni tanlang / Select Language / Выберите язык",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language)
    )

    # Блок с погодой
    with st.expander("🌤️ " + get_text("weather_advice_expander")):
        # Выбор местоположения пользователя
        city_input = st.text_input(get_text("city_input_label"), key="city_input")
        if st.button(get_text("set_location_btn")):
            if city_input.strip():
                place = geocode_city(city_input.strip())
                if place:
                    st.session_state.location = {"lat": place["lat"], "lon": place["lon"]}
                    st.success(f"{get_text('location_set_success')}: {place['name']}")
                else:
                    st.error(get_text("location_not_found"))

        if st.button(get_text("get_weather_btn")):
            weather_data = get_weather(
                st.session_state.location["lat"],
                st.session_state.location["lon"]
            )
            if weather_data:
                temp, humidity, location_name = weather_data
                if location_name:
                    st.write(f"{get_text('weather_location')}: {location_name}")
                st.write(f"{get_text('temperature')}: {temp}°C\n{get_text('humidity')}: {humidity}%")
                risk = get_text("high_risk") if humidity > 80 else get_text("low_risk")
                st.write(f"{get_text('disease_risk')}: {risk}")
            else:
                st.error(get_text("weather_error"))

    # Гид по болезням
    with st.expander(get_text("disease_guide_expander")):
        st.markdown(sidebar_content[st.session_state.language]["disease_guide"])
