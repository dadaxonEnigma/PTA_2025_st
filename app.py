import streamlit as st
import onnxruntime as ort
import numpy as np
import torchvision.models as models
from PIL import Image
import torch
import torch.nn as nn
import cv2
from scipy.special import softmax
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Установка параметров страницы
st.set_page_config(
    page_title="🌿 Plant Disease Detector Pro",
    layout="centered",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# Проверка и импорт Grad-CAM
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    st.error("Библиотека pytorch-grad-cam не установлена. Установите её с помощью: `pip install grad-cam`")
    GRADCAM_AVAILABLE = False

# Заголовок и интерфейс
st.title("🌿 Plant Disease Detector Pro")

# Боковая панель с инструментами
with st.sidebar:
    st.header("🔧 Инструменты")
    
    # Инструмент 1: Справочник болезней
    with st.expander("📚 Справочник болезней", expanded=True):
        st.markdown("""
        - **Чёрная гниль яблони**: Грибковое заболевание, поражающее плоды и листья
        - **Северная пятнистость кукурузы**: Грибковое заболевание листьев
        - **Чёрная гниль винограда**: Опасное грибковое заболевание
        - **Ранняя пятнистость картофеля**: Грибковое заболевание листьев
        """)
    
    # Инструмент 2: Календарь обработки
    with st.expander("📅 Календарь обработки"):
        current_month = datetime.now().strftime("%B")
        st.markdown(f"### Рекомендации на {current_month}")
        st.markdown("""
        - Профилактическая обработка фунгицидами
        - Регулярный осмотр растений
        - Удаление поражённых частей растений
        """)
    
    # Инструмент 3: Калькулятор дозировки
    with st.expander("🧪 Калькулятор дозировки"):
        area = st.number_input("Площадь обработки (м²)", min_value=1, value=100)
        concentration = st.selectbox("Концентрация препарата", ["1%", "2%", "5%"])
        st.markdown(f"**Рекомендуемый объём:** {area * 0.1:.1f} л раствора")
    
    # Инструмент 4: Погодные условия
    with st.expander("⛅ Рекомендации по погоде"):
        st.markdown("""
        - Избегайте обработки перед дождём
        - Оптимальная температура: 15-25°C
        - Лучшее время: утро или вечер
        """)

# Основное содержимое
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "📷 Загрузите фото листа растения", 
        type=["jpg", "jpeg", "png"],
        help="Сфотографируйте поражённый лист на однородном фоне для лучшего результата"
    )

with col2:
    language = st.selectbox("🌐 Выберите язык", ["Русский", "Узбекский", "Английский"])

# Пути к моделям
MODEL_PATH_ONNX = "model/plant_disease_mvp_model.onnx"
MODEL_PATH_PTH = "model/plant_disease_mvp_model.pth"

if not os.path.exists(MODEL_PATH_ONNX) or not os.path.exists(MODEL_PATH_PTH):
    st.error("⚠️ Модель не найдена. Пожалуйста, проверьте путь к файлам модели.")
    st.stop()

# Индикатор загрузки
with st.spinner('🔄 Загрузка модели...'):
    session = ort.InferenceSession(MODEL_PATH_ONNX)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load(MODEL_PATH_PTH, map_location=device))
    model = model.to(device).eval()

classes = [
    'Apple___Black_rot', 'Apple___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___healthy'
]

treatment = {
    'Apple___Black_rot': {
        'description': "Грибковое заболевание, вызываемое Botryosphaeria obtusa",
        'symptoms': "Коричневые пятна на листьях, чёрные гнилые участки на плодах",
        'treatment': "Удалите заражённые плоды и листья. Используйте фунгициды на основе каптана или тиофанат-метила.",
        'prevention': "Регулярная обрезка, уборка опавших листьев, профилактические обработки весной"
    },
    'Apple___healthy': {
        'description': "Здоровое растение без признаков заболеваний",
        'recommendation': "Продолжайте текущий уход. Регулярно осматривайте растение."
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': "Грибковое заболевание, вызываемое Exserohilum turcicum",
        'symptoms': "Длинные серо-зелёные поражения на листьях",
        'treatment': "Используйте устойчивые сорта. Примените фунгициды, такие как азоксистробин.",
        'prevention': "Севооборот, уничтожение растительных остатков"
    },
    'Corn_(maize)___healthy': {
        'description': "Здоровое растение без признаков заболеваний",
        'recommendation': "Продолжайте текущий уход. Регулярно осматривайте растение."
    },
    'Grape___Black_rot': {
        'description': "Грибковое заболевание, вызываемое Guignardia bidwellii",
        'symptoms': "Коричневые пятна с чёрными точками на листьях, сморщенные ягоды",
        'treatment': "Удалите поражённые ягоды и листья. Примените фунгициды на основе каптана.",
        'prevention': "Хорошая вентиляция, профилактические обработки до цветения"
    },
    'Grape___healthy': {
        'description': "Здоровое растение без признаков заболеваний",
        'recommendation': "Продолжайте текущий уход. Регулярно осматривайте растение."
    },
    'Potato___Early_blight': {
        'description': "Грибковое заболевание, вызываемое Alternaria solani",
        'symptoms': "Концентрические кольца на листьях, жёлтые ореолы",
        'treatment': "Примените фунгициды на основе меди. Удалите поражённые листья.",
        'prevention': "Севооборот, избегание переувлажнения"
    },
    'Potato___healthy': {
        'description': "Здоровое растение без признаков заболеваний",
        'recommendation': "Продолжайте текущий уход. Регулярно осматривайте растение."
    }
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_heatmap(image, model, target_class):
    if not GRADCAM_AVAILABLE:
        return None
    try:
        with torch.no_grad():
            input_tensor = preprocess(image).unsqueeze(0).to(device)
        img_np = np.array(image).astype(np.float32) / 255.0
        target_layers = [model.layer4[-1]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        cam_mask = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_mask_resized = cv2.resize(cam_mask, (img_np.shape[1], img_np.shape[0]))
        heatmap = show_cam_on_image(img_np, cam_mask_resized, use_rgb=True)
        return Image.fromarray(heatmap)
    except Exception as e:
        st.error(f"GradCAM ошибка: {e}")
        return None
    
def get_filtered_map(image):
    """Градиент + усиление контраста + яркость"""
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    edges_abs = np.absolute(edges)
    edges_norm = (255 * (edges_abs / np.max(edges_abs))).astype(np.uint8)
    colored = cv2.applyColorMap(edges_norm, cv2.COLORMAP_MAGMA)  # Можно 'JET', 'PLASMA', 'HOT'
    
    # Усиливаем контраст и яркость
    enhanced = cv2.convertScaleAbs(colored, alpha=1.6, beta=30)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(enhanced)
def format_class_name(class_name):
    return class_name.replace("___", " — ").replace("_", " ").title()

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Отображение изображения с возможностью масштабирования
        with st.expander("🔍 Просмотр изображения", expanded=True):
            st.image(img, caption="Загруженное изображение", use_container_width=True)
            
            # Инструменты для работы с изображением
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                rotate = st.button("Повернуть изображение")
            with col_img2:
                download = st.button("Скачать изображение")
            
            if rotate:
                img = img.rotate(90)
                st.experimental_rerun()
            
            if download:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Скачать",
                    data=byte_im,
                    file_name="plant_image.jpg",
                    mime="image/jpeg"
                )

        # Прогресс-бар для визуализации обработки
        progress_bar = st.progress(0)
        
        # Обработка изображения
        img_onnx = img.resize((224, 224))
        img_array = np.array(img_onnx).astype(np.float32) / 255.0
        img_array = img_array.transpose((2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean[:, None, None]) / std[:, None, None]
        img_array = np.expand_dims(img_array, axis=0)
        progress_bar.progress(30)
        
        # Выполнение предсказания
        outputs = session.run([output_name], {input_name: img_array})[0]
        probs = softmax(outputs, axis=1)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        pred_class = classes[top_idx[0]]
        progress_bar.progress(80)
        
        # Отображение результатов
        st.subheader("📊 Результаты диагностики")
        
        # Карточка с основным диагнозом
        with st.container():
            st.markdown(f"### Основной диагноз")
            st.markdown(f"**{format_class_name(pred_class)}** — {probs[top_idx[0]]*100:.1f}%")
            st.markdown(f"**Описание:** {treatment[pred_class]['description']}")
            
            if 'symptoms' in treatment[pred_class]:
                st.markdown(f"### 🔍 Симптомы")
                st.markdown(treatment[pred_class]['symptoms'])
            
            st.markdown(f"### 💡 Рекомендации по лечению")
            st.markdown(treatment[pred_class]['treatment'] if 'treatment' in treatment[pred_class] else treatment[pred_class]['recommendation'])
            
            if 'prevention' in treatment[pred_class]:
                st.markdown(f"**Профилактика:** {treatment[pred_class]['prevention']}")
        
        # Альтернативные варианты диагноза
        with st.expander("🔎 Альтернативные диагнозы", expanded=False):
            for i, idx in enumerate(top_idx[1:]):
                st.write(f"- {format_class_name(classes[idx])}: **{probs[idx]*100:.1f}%**")
        
        progress_bar.progress(100)
        progress_bar.empty()
        
        # Визуализации
        tab1, tab2, tab3, tab4 = st.tabs([
    "📈 График вероятностей", 
    "🌡️ Тепловая карта",
    "🧪 Частотная карта",
    "📋 Полные данные"
])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4' if i == top_idx[0] else '#aec7e8' for i in range(len(probs))]
            ax.barh([format_class_name(classes[i]) for i in range(len(probs))], 
                    probs * 100, 
                    color=colors)
            ax.set_xlabel('Вероятность (%)')
            ax.set_title('Распределение вероятностей по классам')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Кнопка скачивания графика
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            st.download_button(
                label="Скачать график",
                data=buf.getvalue(),
                file_name="disease_probabilities.png",
                mime="image/png"
            )
        
        with tab2:
            if st.button("Сгенерировать тепловую карту"):
                with st.spinner('Создание тепловой карты...'):
                    heatmap_img = get_heatmap(img, model, top_idx[0])
                    if heatmap_img:
                        st.image(
                            heatmap_img, 
                            caption="Тепловая карта (красным выделены области, повлиявшие на диагноз)", 
                            use_container_width=True
                        )
                        
                        # Кнопка скачивания тепловой карты
                        buf = io.BytesIO()
                        heatmap_img.save(buf, format="JPEG")
                        st.download_button(
                            label="Скачать тепловую карту",
                            data=buf.getvalue(),
                            file_name="heatmap.jpg",
                            mime="image/jpeg"
                        )
        with tab3:
            if st.button("Создать частотную карту"):
                with st.spinner("Обработка изображения..."):
                    filtered_img = get_filtered_map(img)
                    st.image(filtered_img, caption="Карта градиентов (частотных изменений)", use_container_width=True)
                    
                    buf = io.BytesIO()
                    filtered_img.save(buf, format="JPEG")
                    st.download_button(
                        label="Скачать частотную карту",
                        data=buf.getvalue(),
                        file_name="frequency_map.jpg",
                        mime="image/jpeg"
                    )
                
        with tab4:
            st.json({
                "diagnosis": format_class_name(pred_class),
                "probability": f"{probs[top_idx[0]]*100:.2f}%",
                "timestamp": datetime.now().isoformat(),
                "all_probabilities": {
                    format_class_name(classes[i]): f"{probs[i]*100:.2f}%" 
                    for i in range(len(probs))
                }
            })
            
            # Кнопка скачивания полного отчёта
            report = f"""
            Отчёт о диагностике растения
            ===========================
            
            Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            Основной диагноз: {format_class_name(pred_class)}
            Вероятность: {probs[top_idx[0]]*100:.1f}%
            
            Все вероятности:
            {chr(10).join([f"- {format_class_name(classes[i])}: {probs[i]*100:.2f}%" for i in range(len(probs))])}
            
            Рекомендации:
            {treatment[pred_class]['treatment'] if 'treatment' in treatment[pred_class] else treatment[pred_class]['recommendation']}
            """
            
            st.download_button(
                label="Скачать полный отчёт",
                data=report,
                file_name="plant_diagnosis_report.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")
else:
    # Демонстрационный раздел с примерами
    st.markdown("### Как пользоваться приложением?")
    st.markdown("""
    1. Сфотографируйте поражённый лист растения на однородном фоне
    2. Загрузите изображение в поле выше
    3. Получите диагноз и рекомендации по лечению
    
    **Совет:** Для лучших результатов убедитесь, что лист хорошо освещён и занимает большую часть кадра.
    """)
    
    # Примеры изображений
    st.subheader("📌 Примеры изображений для анализа")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        st.image("https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Apple___Black_rot/0a4c0e4b-2ce3-4c1e-9bd6-4f4d0a1e57c3___JR_FrgE.S 8580.JPG", 
                caption="Чёрная гниль яблони", use_container_width=True)
    with col_ex2:
        st.image("https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Corn_(maize)___Northern_Leaf_Blight/0bfa3614-faee-4fbf-92aa-1c2a9f1d0476___RS_NLB 4023.JPG", 
                caption="Северная пятнистость кукурузы", use_container_width=True)
    with col_ex3:
        st.image("https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Potato___Early_blight/0a2ca798-8777-4d98-ab05-3a7d1b6f14a8___RS_Erly.B 8432.JPG", 
                caption="Ранняя пятнистость картофеля", use_container_width=True)