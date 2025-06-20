import streamlit as st
from PIL import Image
from datetime import datetime
from models.inference import load_models, predict_disease
from models.visualization import get_heatmap, get_filtered_map, plot_probabilities
from web_search.search import web_search
from data.treatments import get_treatment
from data.classes import classes
from data.translations import translations, class_name_translations, sidebar_content
import config
import io

# Инициализация состояния языка
if 'language' not in st.session_state:
    st.session_state.language = "uz"

# Функция для получения переводов
def get_text(key, lang=None):
    if lang is None:
        lang = st.session_state.language
    return translations[lang].get(key, key)

# Настройка страницы
st.set_page_config(
    page_title=get_text("page_title"),
    layout="centered",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# Подключение CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def initialize_app():
    try:
        return load_models()
    except FileNotFoundError as e:
        st.error(f"⚠️ {str(e)}")
        st.stop()

session, model = initialize_app()

# Проверка словаря treatment
treatment = get_treatment(st.session_state.language)
for cls in classes:
    if cls not in treatment:
        st.error(f"Xato: '{cls}' klassi data/treatments.py da topilmadi.")
        st.stop()
    if 'treatment' not in treatment[cls] and 'recommendation' not in treatment[cls]:
        st.error(f"Xato: '{cls}' uchun 'treatment' yoki 'recommendation' topilmadi.")
        st.stop()



# Боковая панель
# Фрагмент для боковой панели в app.py
with st.sidebar:
    st.header(get_text("sidebar_tools"))
    # Выбор языка с эмодзи флагов
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
    
    with st.expander(get_text("disease_guide_expander"), expanded=True):
        st.markdown(sidebar_content[st.session_state.language]["disease_guide"])
    with st.expander(get_text("treatment_schedule_expander")):
        st.markdown(f"### {get_text('treatment_schedule_title').format(month=datetime.now().strftime('%B'))}")
        st.markdown(sidebar_content[st.session_state.language]["treatment_schedule"])
    with st.expander(get_text("weather_advice_expander")):
        st.markdown(sidebar_content[st.session_state.language]["weather_advice"])

st.title(get_text("page_title"))
uploaded_file = st.file_uploader(
    get_text("file_uploader_label"),
    type=["jpg", "jpeg", "png"],
    help=get_text("file_uploader_help")
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    with st.expander(get_text("image_view_expander"), expanded=True):
        st.image(img, caption=get_text("image_caption"), use_container_width=True)

    with st.spinner(get_text("processing_message")):
        pred_class, probs, top_idx = predict_disease(img, session, model, classes)
    
    st.subheader(get_text("results_header"))
    with st.container():
        st.markdown(f"### {get_text('main_diagnosis')}")
        st.markdown(f"**{config.format_class_name(pred_class, st.session_state.language)}** — {probs[top_idx[0]]*100:.1f}%")
        st.markdown(f"**{get_text('description_label')}:** {treatment[pred_class]['description']}")
        if 'symptoms' in treatment[pred_class]:
            st.markdown(f"### {get_text('symptoms_label')}")
            st.markdown(treatment[pred_class]['symptoms'])
        st.markdown(f"### {get_text('treatment_label')}")
        st.markdown(treatment[pred_class].get('treatment', treatment[pred_class].get('recommendation', get_text('no_treatment'))))
        if 'prevention' in treatment[pred_class]:
            st.markdown(f"**{get_text('prevention_label')}:** {treatment[pred_class]['prevention']}")

    with st.expander(get_text("alternative_diagnoses"), expanded=False):
        for i, idx in enumerate(top_idx[1:]):
            st.write(f"- {config.format_class_name(classes[idx], st.session_state.language)}: **{probs[idx]*100:.1f}%**")

    # Визуализации
    tabs = st.tabs(get_text("visualization_tabs"))
    
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
    
    with tabs[1]:
        if st.button(get_text("generate_heatmap")):
            with st.spinner(get_text('processing_message')):
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
    
    with tabs[2]:
        if st.button(get_text("generate_freq_map")):
            with st.spinner(get_text("processing_message")):
                filtered_img = get_filtered_map(img)
                st.image(filtered_img, caption=get_text("freq_map_caption"), use_container_width=True)
                buf = io.BytesIO()
                filtered_img.save(buf, format="JPEG")
                st.download_button(
                    label=get_text("download_freq_map"),
                    data=buf.getvalue(),
                    file_name="frequency_map.jpg",
                    mime="image/jpeg"
                )
    
    with tabs[3]:
        report_data = {
            "tashxis": config.format_class_name(pred_class, st.session_state.language),
            "ehtimollik": f"{probs[top_idx[0]]*100:.2f}%",
            "vaqt": datetime.now().isoformat(),
            "barcha_ehtimolliklar": {
                config.format_class_name(classes[i], st.session_state.language): f"{probs[i]*100:.2f}%"
                for i in range(len(probs))
            }
        }
        st.json(report_data)
        report = get_text("report_template").format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            diagnosis=config.format_class_name(pred_class, st.session_state.language),
            probability=probs[top_idx[0]]*100,
            probabilities=chr(10).join([f"- {config.format_class_name(classes[i], st.session_state.language)}: {probs[i]*100:.2f}%" for i in range(len(probs))]),
            treatment=treatment[pred_class].get('treatment', treatment[pred_class].get('recommendation', get_text('no_treatment')))
        )
        st.download_button(
            label=get_text("full_report_label"),
            data=report,
            file_name="plant_diagnosis_report.txt",
            mime="text/plain"
        )
    
    with tabs[4]:
        st.markdown(f"### {get_text('web_search_header')}")
        if st.button(get_text("web_search_label")):
            with st.spinner(get_text("chat_processing")):
                web_results = web_search(f"{config.format_class_name(pred_class, st.session_state.language)} kasalligi davolash")
                if web_results and "error" not in web_results[0]:
                    for result in web_results:
                        st.markdown(f"**{result['title']}**")
                        st.markdown(f"{result['description']}")
                        st.markdown(f"[Manbaga o‘tish]({result['url']})")
                        st.markdown("---")
                else:
                    st.error(get_text("chat_web_error"))

    # Чат-бот
    st.markdown("---")
    st.subheader(get_text("chat_header"))
    with st.form("chat_form"):
        user_input = st.text_input(get_text("chat_input_label"))
        submitted = st.form_submit_button(get_text("chat_submit_button"))
        if submitted and user_input.strip():
            with st.spinner(get_text("chat_processing")):
                web_results = web_search(f"{user_input} o‘simlik kasalligi")
                if web_results and "error" not in web_results[0]:
                    for result in web_results:
                        st.markdown(f"✅ **{result['title']}**")
                        st.markdown(f"{result['description']}")
                        st.markdown(f"[📖 Manbaga o‘tish]({result['url']})")
                        st.markdown("---")
                else:
                    st.error(get_text("chat_web_error"))
else:
    st.markdown(get_text("usage_guide"))
    st.subheader(get_text("sample_images_header"))
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    for col, img_url, cls in [
        (col_ex1, config.SAMPLE_IMAGES[0], "Apple___Black_rot"),
        (col_ex2, config.SAMPLE_IMAGES[1], "Corn_(maize)___Northern_Leaf_Blight"),
        (col_ex3, config.SAMPLE_IMAGES[2], "Potato___Early_blight")
    ]:
        with col:
            st.image(img_url, caption=config.format_class_name(cls, st.session_state.language), use_container_width=True)