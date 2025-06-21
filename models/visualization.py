# models/visualization.py
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

import config
from data.translations import translations


def get_heatmap(image, model, target_class):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        st.error("Библиотека pytorch-grad-cam не установлена. Установите её: `pip install grad-cam`")
        return None
    
    try:
        with torch.no_grad():
            input_tensor = config.preprocess(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        img_np = np.array(image).astype(np.float32) / 255.0
        target_layers = [model.layer4[-1]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        cam_mask = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_mask_resized = cv2.resize(cam_mask, (img_np.shape[1], img_np.shape[0]))
        heatmap = show_cam_on_image(img_np, cam_mask_resized, use_rgb=True)
        return Image.fromarray(heatmap)
    except Exception as e:
        st.error(f"GradCAM xatosi: {e}")
        return None

def get_filtered_map(image, filter_type='sobel'):
    img_np = np.array(image)
    if filter_type == 'sobel':
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        edges_abs = np.absolute(edges)
        edges_norm = (255 * (edges_abs / np.max(edges_abs))).astype(np.uint8)
        colored = cv2.applyColorMap(edges_norm, cv2.COLORMAP_MAGMA)
        enhanced = cv2.convertScaleAbs(colored, alpha=1.6, beta=30)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced)
    elif filter_type == 'gray':
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(gray)
    elif filter_type == 'canny':
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges)
    else:
        # Возвращаем исходное изображение, если фильтр неизвестен
        return image

def plot_probabilities(probs, classes, top_idx):
    lang = st.session_state.get('language', 'uz')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4' if i == top_idx[0] else '#aec7e8' for i in range(len(probs))]
    ax.barh([config.format_class_name(classes[i], lang) for i in range(len(probs))], 
            probs * 100, 
            color=colors)
    ax.set_xlabel(translations[lang]["visualization_tabs"][0].split()[1])  # "Ehtimollik (%)" или "Probability (%)"
    ax.set_title(translations[lang]["visualization_tabs"][0])
    plt.tight_layout()
    return fig