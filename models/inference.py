# models/inference.py
import onnxruntime as ort
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from PIL import Image
import config
import os

def load_models():
    print(f"ONNX Path: {config.MODEL_PATH_ONNX}, Exists: {os.path.exists(config.MODEL_PATH_ONNX)}")
    print(f"PTH Path: {config.MODEL_PATH_PTH}, Exists: {os.path.exists(config.MODEL_PATH_PTH)}")
    if not os.path.exists(config.MODEL_PATH_ONNX) or not os.path.exists(config.MODEL_PATH_PTH):
        raise FileNotFoundError("Model topilmadi. Model fayllari yo‘lini tekshiring.")
    
    session = ort.InferenceSession(config.MODEL_PATH_ONNX)
    print(f"ONNX Input: {session.get_inputs()[0].name}, Output: {session.get_outputs()[0].name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load(config.MODEL_PATH_PTH, map_location=device))
    model = model.to(device).eval()
    return session, model

def predict_disease(img, session, model, classes):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    input_tensor = config.preprocess(img).unsqueeze(0).numpy()
    
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    probs = softmax(outputs, axis=1)[0]
    top_idx = np.argsort(probs)[::-1][:3]
    pred_class = classes[top_idx[0]]
    
    return pred_class, probs, top_idx