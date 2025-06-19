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

# Проверка и импорт Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    st.error("Библиотека pytorch-grad-cam не установлена. Установите её с помощью: `pip install grad-cam`")
    GRADCAM_AVAILABLE = False

# Настройка страницы
st.set_page_config(page_title="🧠 Plant Disease Detector", layout="centered")
st.title("🌿 Определение болезни растения по фото")
st.write("Загрузите изображение листа, и модель определит болезнь 🌱")

# Пути к моделям
MODEL_PATH_ONNX = "model/plant_disease_mvp_model.onnx"
MODEL_PATH_PTH = "model/plant_disease_mvp_model.pth"

# Проверка наличия моделей
if not os.path.exists(MODEL_PATH_ONNX) or not os.path.exists(MODEL_PATH_PTH):
    st.error(f"Модель не найдена: {MODEL_PATH_ONNX} или {MODEL_PATH_PTH}")
    st.stop()

# Загрузка ONNX модели
session = ort.InferenceSession(MODEL_PATH_ONNX)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Загрузка PyTorch модели для Grad-CAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 8)  # 8 классов
model.load_state_dict(torch.load(MODEL_PATH_PTH, map_location=device))
model = model.to(device).eval()

# Список классов
classes =  ['Apple___Black_rot', 
            'Apple___healthy', 
            'Corn_(maize)___Northern_Leaf_Blight', 
            'Corn_(maize)___healthy', 
            'Grape___Black_rot', 
            'Grape___healthy', 
            'Potato___Early_blight', 
            'Potato___healthy']

# Рекомендации по лечению
treatment = {
    'Apple___Black_rot': "Удалите заражённые плоды и листья, используйте фунгициды на основе каптана или тиофанат-метила.",
    'Apple___healthy': "Растение здорово 🍎",
    
    'Corn_(maize)___Northern_Leaf_Blight': "Используйте устойчивые сорта, примените фунгициды, такие как азоксистробин.",
    'Corn_(maize)___healthy': "Растение здорово 🌽",


    'Grape___Black_rot': "Удалите поражённые ягоды и листья, примените фунгициды на основе каптана.",
    'Grape___healthy': "Растение здорово 🍇",

    'Potato___Early_blight': "Примените фунгициды на основе меди, удалите поражённые листья.",
    'Potato___healthy': "Растение здорово 🥔",
    
}


# Преобразования
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для тепловой карты
def get_heatmap(image, model, target_class):
    if not GRADCAM_AVAILABLE:
        return None
    try:
        model.eval()
        img = np.array(image).astype(np.float32) / 255.0
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        cam_mask = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_mask_resized = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
        heatmap = show_cam_on_image(img, cam_mask_resized, use_rgb=True)
        return Image.fromarray(heatmap)
    except Exception as e:
        st.error(f"GradCAM ошибка: {e}")
        return None

# Загрузка изображения
uploaded_file = st.file_uploader("📷 Загрузите изображение", type=["jpg", "jpeg", "png"])

# Обработка изображения
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Загруженное изображение", use_container_width=True)

        # Предобработка для ONNX
        img_onnx = img.resize((224, 224))
        img_array = np.array(img_onnx).astype(np.float32) / 255.0
        img_array = img_array.transpose((2, 0, 1))  # HWC -> CHW
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean[:, None, None]) / std[:, None, None]
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание с ONNX
        outputs = session.run([output_name], {input_name: img_array})[0]
        probs = softmax(outputs, axis=1)[0]
        top_idx = np.argsort(probs)[::-1][:3]

        # Вывод результатов
        st.subheader("📊 Результаты:")
        for i, idx in enumerate(top_idx):
            label = classes[idx].replace("___", " — ").replace("_", " ").title()
            st.write(f"- {i+1}. {label}: **{probs[idx]*100:.1f}%**")

        pred_class = classes[top_idx[0]]
        st.success(f"✅ Вероятнее всего: *{pred_class.replace('___', ' — ').replace('_', ' ').title()}*")
        st.info(f"💡 Рекомендация: {treatment.get(pred_class, 'Информация отсутствует')}")

        # Кнопка для тепловой карты
        if st.button("📈 Показать тепловую карту (почему так определено)"):
            heatmap_img = get_heatmap(img, model, top_idx[0])
            if heatmap_img:
                st.image(heatmap_img, caption="Тепловая карта (области, повлиявшие на предсказание)", use_container_width=True)
                st.write("Тепловая карта показывает, какие области изображения повлияли на решение модели.")

        # График вероятностей
        st.subheader("📈 График вероятностей")
        fig, ax = plt.subplots()
        ax.barh([classes[i].replace("___", " — ").title() for i in range(len(probs))], probs * 100, color='skyblue')
        ax.set_xlabel('Вероятность (%)')
        ax.set_title('Вероятности классов')
        plt.tight_layout()
        st.pyplot(fig)

        # Логирование всех вероятностей
        st.write("Все вероятности:", {classes[i].replace("___", " — ").replace("_", " ").title(): f"{probs[i]*100:.2f}%" for i in range(len(probs))})

    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {e}")