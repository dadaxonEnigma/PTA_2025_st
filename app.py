import streamlit as st
from PIL import Image
from ui.chat import render_chat
from models.inference import load_models
from ui.diagnosis import render_diagnosis
from ui.sidebar import render_sidebar
from data.treatments import get_treatment
from utils.weather import get_weather
from data.classes import classes
from utils.i18n import get_text
import config
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

# --- Инициализация сессии и API ---
api_key = st.secrets["API_KEY"]
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

if 'location' not in st.session_state:
    st.session_state.location = {"lat": 41.3, "lon": 69.2}  # Пример для Ташкента

# Инициализация языка и чата
if 'language' not in st.session_state:
    st.session_state.language = "uz"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_context' not in st.session_state:  # NEW: Контекст чата
    st.session_state.chat_context = ""


treatment = get_treatment(st.session_state.language)

# --- Настройка страницы ---
st.set_page_config(
    page_title=get_text("page_title"),
    layout="centered",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Загрузка моделей ---
@st.cache_resource(ttl=3600)  # NEW: Кеширование на 1 час
def initialize_app():
    try:
        return load_models()
    except FileNotFoundError as e:
        st.error(f"⚠️ {str(e)}")
        st.stop()

session, model = initialize_app()

# --- Боковая панель ---
with st.sidebar:
    render_sidebar(get_text, get_weather, treatment)
# --- Основной интерфейс ---
st.title(get_text("page_title"))
uploaded_file = st.file_uploader(
    get_text("file_uploader_label"),
    type=["jpg", "jpeg", "png"],
    help=get_text("file_uploader_help")
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    pred_class, probs, top_idx = render_diagnosis(
        img, session, model, classes, treatment, get_text, config
    )
    st.session_state['pred_class'] = pred_class

    # --- Чат с DeepSeek ---
    render_chat(pred_class, probs, treatment[pred_class], api_key, get_text, config)


# NEW: Блок с примерами изображений
else:
    st.markdown(get_text("usage_guide"))
    st.subheader(get_text("sample_images_header"))
    cols = st.columns(3)
    for col, img_url, cls in zip(
        cols,
        config.SAMPLE_IMAGES,
        ["Apple___Black_rot", "Corn_(maize)___Northern_Leaf_Blight", "Potato___Early_blight"]
    ):
        with col:
            st.image(img_url, caption=config.format_class_name(cls, st.session_state.language), use_container_width=True)