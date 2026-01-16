#!/bin/bash

# Скрипт автоматической установки для MacBook
# Запуск: bash setup_mac.sh

echo "🍎 Установка Plant Disease Detection для MacBook"
echo "================================================"
echo ""

# Проверка Python
echo "📋 Шаг 1/6: Проверка Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python не найден!"
    echo "Установите Python: brew install python@3.11"
    echo "Или скачайте с https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ Python найден: $PYTHON_VERSION"
echo ""

# Создание виртуального окружения
echo "📋 Шаг 2/6: Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo "⚠️  venv уже существует, пропускаем..."
else
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
fi
echo ""

# Активация виртуального окружения
echo "📋 Шаг 3/6: Активация виртуального окружения..."
source venv/bin/activate
echo "✅ Виртуальное окружение активировано"
echo ""

# Обновление pip
echo "📋 Шаг 4/6: Обновление pip..."
pip install --upgrade pip --quiet
echo "✅ pip обновлен"
echo ""

# Установка зависимостей
echo "📋 Шаг 5/6: Установка зависимостей..."
echo "⏱️  Это может занять 2-5 минут..."
if [ -f "requirements_mac.txt" ]; then
    pip install -r requirements_mac.txt --quiet
    echo "✅ Зависимости установлены (requirements_mac.txt)"
else
    echo "⚠️  requirements_mac.txt не найден, устанавливаю базовые пакеты..."
    pip install streamlit pillow requests numpy onnxruntime --quiet
    echo "✅ Базовые зависимости установлены"
fi
echo ""


# Проверка модели
echo "📋 Проверка модели..."
if [ ! -f "models/plant_disease_model.onnx" ]; then
    echo "⚠️  Модель не найдена: models/plant_disease_model.onnx"
    echo "❗ Вам нужно добавить файл модели в папку models/"
    echo ""
    echo "Создать заглушку для тестирования? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        mkdir -p models
        touch models/plant_disease_model.onnx
        echo "✅ Создана заглушка модели (только для тестирования)"
    fi
else
    echo "✅ Модель найдена"
fi
echo ""

# Финальные инструкции
echo "================================================"
echo "🎉 Установка завершена!"
echo "================================================"
echo ""
echo "Следующие шаги:"
echo "Для запуска приложения:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Для остановки: Ctrl + C"
echo ""
echo "✨ Удачи!"