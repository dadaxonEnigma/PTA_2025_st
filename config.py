import torchvision.transforms as transforms
from data.translations import class_name_translations

# Пути к моделям
MODEL_PATH_ONNX = "model/plant_disease_mvp_model.onnx"
MODEL_PATH_PTH = "model/plant_disease_mvp_model.pth"

# Настройки интерфейса
LAYOUT = "centered"
PAGE_ICON = "🌱"
SIDEBAR_STATE = "expanded"

# Ссылки на образцы изображений
SAMPLE_IMAGES = [
    "https://gardenerspath.com/wp-content/uploads/2021/03/Frogeye-Leaf-Spot-aka-Black-Rot-Botryosphaeria-on-Apple-Tree.jpg",
    "https://assets.syngentaebiz.com/images/GrayLeafSpotLow-Res.jpg",
    "https://kj2bcdn.b-cdn.net/media/30352/early-blight-of-potato.jpeg"
]

# Функция для форматирования названий классов с учётом языка
def format_class_name(class_name, lang="uz"):
    return class_name_translations[lang].get(class_name, class_name.replace("___", " — ").replace("_", " ").title())

# Предобработка изображений
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])