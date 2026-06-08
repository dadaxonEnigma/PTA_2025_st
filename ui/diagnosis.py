import streamlit as st
import io

from data.translations import class_name_translations
from web_search.search import web_search
from utils.weather import get_weather, get_forecast
import config
from models.visualization import (
    plot_probabilities,
    get_heatmap,
    get_filtered_map
)

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def get_en_search_query(pred_class: str) -> str:
    """
    Формирует КАЧЕСТВЕННЫЙ EN-запрос для поиска
    """
    try:
        # Например: "Corn Northern Leaf Blight"
        full_name = class_name_translations["en"][pred_class]
    except KeyError:
        # fallback
        full_name = pred_class.replace("___", " ").replace("_", " ")

    query = (
        f"{full_name} "
        f"treatment symptoms fungicide management "
        f"site:.edu OR site:.gov OR site:.org"
    )

    return query


# =========================================================
# ОСНОВНОЙ РЕНДЕР ДИАГНОЗА
# =========================================================

def render_diagnosis(img, session, model, classes, treatment, get_text, config):
    with st.expander(get_text("image_view_expander"), expanded=True):
        st.image(img, caption=get_text("image_caption"), use_container_width=True)

    with st.spinner(get_text("processing_message")):
        pred_class, probs, top_idx = model_predict(img, session, model, classes)

    # сохраняем диагноз
    st.session_state.pred_class = pred_class

    st.subheader(get_text("results_header"))

    with st.container():
        st.markdown(f"### {get_text('main_diagnosis')}")

        st.markdown(
            f"**{config.format_class_name(pred_class, st.session_state.language)}** "
            f"— {probs[top_idx[0]] * 100:.1f}%"
        )

        st.markdown(
            f"**{get_text('description_label')}:** "
            f"{treatment[pred_class]['description']}"
        )

        if 'symptoms' in treatment[pred_class]:
            st.markdown(f"### {get_text('symptoms_label')}")
            st.markdown(treatment[pred_class]['symptoms'])

        st.markdown(f"### {get_text('treatment_label')}")
        st.markdown(
            treatment[pred_class].get(
                'treatment',
                treatment[pred_class].get(
                    'recommendation',
                    get_text('no_treatment')
                )
            )
        )

        if 'prevention' in treatment[pred_class]:
            st.markdown(
                f"**{get_text('prevention_label')}:** "
                f"{treatment[pred_class]['prevention']}"
            )

        render_weather_risk(pred_class, get_text)
        render_forecast_risk(pred_class, probs, top_idx, get_text)

    render_visualizations(img, model, probs, top_idx, classes, get_text)

    return pred_class, probs, top_idx


# =========================================================
# ПОГОДА И РИСК РАСПРОСТРАНЕНИЯ БОЛЕЗНИ
# =========================================================

def render_weather_risk(pred_class, get_text):
    st.markdown(f"### {get_text('weather_risk_header')}")

    weather_data = get_weather(
        st.session_state.location["lat"],
        st.session_state.location["lon"]
    )

    if not weather_data:
        st.info(get_text("weather_unavailable"))
        return

    temp, humidity, location_name = weather_data
    disease_name = config.format_class_name(pred_class, st.session_state.language)

    if location_name:
        st.write(f"{get_text('weather_location')}: {location_name}")
    st.write(f"{get_text('weather_temp')}: {temp}°C")
    st.write(f"{get_text('weather_humidity')}: {humidity}%")

    risk_key = "weather_risk_high" if humidity > 80 else "weather_risk_low"
    risk_text = get_text(risk_key).format(humidity=humidity, temp=temp, disease=disease_name)

    if humidity > 80:
        st.warning(risk_text)
    else:
        st.success(risk_text)


def render_forecast_risk(pred_class, probs, top_idx, get_text):
    st.markdown(f"### {get_text('forecast_header')}")

    forecast = get_forecast(
        st.session_state.location["lat"],
        st.session_state.location["lon"]
    )

    if not forecast:
        st.info(get_text("weather_unavailable"))
        return

    dates = [day["date"] for day in forecast]
    selected_date = st.selectbox(
        get_text("forecast_select_date"),
        dates,
        key="forecast_date"
    )

    day = next(d for d in forecast if d["date"] == selected_date)
    disease_name = config.format_class_name(pred_class, st.session_state.language)
    confidence = probs[top_idx[0]] * 100

    st.write(f"{get_text('weather_temp')}: {day['temp']}°C")
    st.write(f"{get_text('weather_humidity')}: {day['humidity']}%")

    risk_key = "forecast_risk_high" if day["humidity"] > 80 else "forecast_risk_low"
    risk_text = get_text(risk_key).format(
        date=selected_date,
        humidity=day["humidity"],
        temp=day["temp"],
        disease=disease_name,
        confidence=f"{confidence:.1f}"
    )

    if day["humidity"] > 80:
        st.warning(risk_text)
    else:
        st.success(risk_text)


# =========================================================
# ИНФЕРЕНС
# =========================================================

def model_predict(img, session, model, classes):
    from models.inference import predict_disease
    return predict_disease(img, session, model, classes)


# =========================================================
# ВИЗУАЛИЗАЦИИ + ВЕБ-ПОИСК
# =========================================================

def render_visualizations(img, model, probs, top_idx, classes, get_text):
    tabs = st.tabs(get_text("visualization_tabs"))

    # --- Таб 1: График ---
    with tabs[0]:
        fig = plot_probabilities(probs, classes, top_idx)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        st.download_button(
            label=get_text("download_graph"),
            data=buf.getvalue(),
            file_name="disease_probabilities.png",
            mime="image/png"
        )

    # --- Таб 2: Тепловая карта ---
    with tabs[1]:
        if st.button(get_text("generate_heatmap")):
            with st.spinner(get_text("processing_message")):
                heatmap_img = get_heatmap(img, model, top_idx[0])
                if heatmap_img:
                    st.image(
                        heatmap_img,
                        caption=get_text("heatmap_caption"),
                        use_container_width=True
                    )

                    buf = io.BytesIO()
                    heatmap_img.save(buf, format="JPEG")
                    st.download_button(
                        label=get_text("download_heatmap"),
                        data=buf.getvalue(),
                        file_name="heatmap.jpg",
                        mime="image/jpeg"
                    )

    # --- Таб 3: Частотная карта ---
    with tabs[2]:
        if st.button(get_text("generate_freq_map")):
            with st.spinner(get_text("processing_message")):
                filtered_img = get_filtered_map(img)
                st.image(
                    filtered_img,
                    caption=get_text("freq_map_caption"),
                    use_container_width=True
                )

                buf = io.BytesIO()
                filtered_img.save(buf, format="JPEG")
                st.download_button(
                    label=get_text("download_freq_map"),
                    data=buf.getvalue(),
                    file_name="frequency_map.jpg",
                    mime="image/jpeg"
                )

    # =====================================================
    # Таб 4: ВЕБ-ПОИСК (EN ПОИСК → RU ИНТЕРФЕЙС)
    # =====================================================

    with tabs[3]:
        st.markdown(f"### {get_text('web_search_header')}")

        if st.button(get_text("web_search_label")):
            if 'pred_class' not in st.session_state:
                st.warning("Сначала сделайте предсказание!")
            else:
                pred_class = st.session_state.pred_class

                query = get_en_search_query(pred_class)

                # показываем пользователю ЧТО ищем (нормально для дебага)
                st.caption(f"Search query (EN): {query}")

                with st.spinner(get_text("chat_processing")):
                    st.session_state.web_results = web_search(
                        query,
                        max_results=10
                    )

        web_results = st.session_state.get("web_results", [])

        if web_results:
            for result in web_results:
                st.markdown(f"**{result['title']}**")
                st.markdown(result['description'])
                st.markdown(
                    f"[{get_text('source_link')}]({result['url']})"
                )
                st.markdown("---")
        elif 'web_results' in st.session_state:
            st.error(get_text("chat_web_error"))
