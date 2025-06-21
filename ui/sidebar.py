import streamlit as st
from data.translations import sidebar_content

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
        if st.button(get_text("get_weather_btn")):
            temp_humidity = get_weather(
                st.session_state.location["lat"],
                st.session_state.location["lon"]
            )
            if temp_humidity:
                temp, humidity = temp_humidity
                st.write(f"{get_text('temperature')}: {temp}°C\n{get_text('humidity')}: {humidity}%")
                risk = get_text("high_risk") if humidity > 80 else get_text("low_risk")
                st.write(f"{get_text('disease_risk')}: {risk}")
            else:
                st.error(get_text("weather_error"))

    # Гид по болезням
    with st.expander(get_text("disease_guide_expander")):
        st.markdown(sidebar_content[st.session_state.language]["disease_guide"])
