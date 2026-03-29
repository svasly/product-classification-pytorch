# Product Classification with PyTorch

Классификация товаров по изображениям с использованием глубокого обучения.

## Описание

Проект для автоматической категоризации товаров маркетплейса на основе фотографий. Система классифицирует изображения по 10 категориям FashionMNIST с точностью **94.26%** на тестовом наборе и успешно работает на реальных фотографиях товаров.

## Результаты

### На тестовом наборе (FashionMNIST)

| Метрика | Значение |
|---------|----------|
| **Accuracy** | **94.26%** |
| Best Class (F1) | Trouser (0.994) |
| Worst Class (F1) | Shirt (0.829) |

### На реальных изображениях

| Изображение | Предсказание | Уверенность |
|-------------|--------------|-------------|
| bag.jpg | Bag | 99.98% |
| t-shirt.jpg | T-shirt/top | 99.85% |
| sneaker.jpg | Sneaker | 98.48% |
| dress.jpg | Dress | 70.28% |

## Возможности

- **Transfer Learning**: ResNet18, предобученная на ImageNet
- **Адаптация архитектуры**: под одноканальные изображения 28x28
- **Умная предобработка**: автоматическая инверсия, crop и padding для реальных фото
- **Стабильное обучение**: градиентный клиппинг, early stopping
- **Поддержка GPU**: CUDA, MPS (Apple Silicon) и CPU

## Архитектура модели

- **Backbone**: ResNet18 (предобученная)
- **Вход**: 1 канал (grayscale), 28x28 пикселей
- **Модификации**:
  - Первый conv слой адаптирован под 1 канал (суммирование весов RGB)
  - Stride первого слоя: 1 (вместо 2 для сохранения разрешения)
  - MaxPool заменён на Identity
  - Последний FC слой: 512 → 10 классов
- **Оптимизатор**: AdamW (lr=1e-3, weight_decay=1e-2)
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)

## Структура проекта
``
product-classification/
├── data/ # Датасеты (скачиваются автоматически)
├── src/
│ ├── init.py
│ ├── dataset.py # Загрузка и аугментация данных
│ ├── model.py # Архитектура нейросети
│ ├── train.py # Цикл обучения с валидацией
│ ├── evaluate.py # Оценка на тестовом наборе
│ ├── predict.py # Инференс на новых изображениях
│ └── plot_history.py # Визуализация метрик обучения
├── notebooks/ # Jupyter ноутбуки для экспериментов
├── best_model.pth # Веса лучшей модели
├── history.json # История обучения
├── evaluation_results.json # Полные метрики оценки
├── confusion_matrix.png # Матрица ошибок
├── training_plots.png # Графики обучения
├── debug_processed.png # Обработанное изображение (для отладки)
├── requirements.txt # Зависимости
└── README.md
``
## Требования

- Python: 3.12+
- CUDA Toolkit: 12.4 (опционально, для GPU)
- PyTorch: 2.0+

## Установка

### 1. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Для Linux/macOS
# или
venv\Scripts\activate     # Для Windows

### 2. Установка PyTorch с поддержкой CUDA**

**2.1. Проверяем поддерживаемую версию:**
```
nvidia-smi
```
**2.2. Устанавливаем стабильную версию:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
**2.3. Проверка установки:**
```
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"
```
### 3. Установка остальных зависимостей:
```
pip install -r requirements.txt
```

## Использование
### Обучение модели
```
python src/train.py
```

**Параметры (в src/train.py):**

- BATCH_SIZE (по умолчанию 64)
- NUM_EPOCHS (по умолчанию 15)
- LEARNING_RATE (по умолчанию 1e-3)

**Результаты обучения:**

- best_model.pth — веса лучшей модели
- history.json — метрики по эпохам

## Оценка модели
```
python src/evaluate.py
```
**Генерирует:**

- evaluation_results.json — полные метрики (precision, recall, F1)
- confusion_matrix.png — визуализация матрицы ошибок

## Предсказание на новых изображениях
```
python src/predict.py <путь_к_изображению> [путь_к_модели]
```
**Примеры:**

```
# Предсказание с моделью по умолчанию
python src/predict.py sneaker.jpg

# С указанием конкретной модели
python src/predict.py dress.jpg best_model.pth
```
## Особенности предобработки:

- Автоматическая инверсия цветов (для соответствия FashionMNIST)
- Обрезка лишнего фона (auto-crop)
- Добавление полей (padding)
- Сохранение debug_processed.png для проверки

## Пример вывода:
```
Using device: cuda
Loading model from best_model.pth...
Model loaded successfully!

Processing image: sneaker.jpg
Detected light background -> Inverting to black

==================================================
PREDICTION RESULT
==================================================
Predicted class: Sneaker
Confidence: 98.48%

Top 3 predictions:
--------------------------------------------------
  Sneaker         98.48%
  Ankle boot      1.16%
  Sandal          0.25%
==================================================

Debug: Processed image saved to 'debug_processed.png'
```
## Визуализация истории обучения
```
python src/plot_history.py
```
**Генерирует: `training_plots.png` с графиками `Loss` и `Accuracy`**

## Воспроизводимость

**Для детерминированных результатов:**

- torch.manual_seed(42)
- torch.cuda.manual_seed_all(42)

Результаты могут незначительно отличаться из-за недетерминированных операций в CUDA.

## Возможности улучшения

1. Data Augmentation: Добавить ColorJitter, RandomCrop, RandomRotation
2. Fine-tuning: Разморозить верхние слои ResNet
3. Ensemble: Объединить несколько моделей
4. Architecture: Попробовать EfficientNet или MobileNet
5. Dataset: Обучить на полном FashionMNIST с большим количеством аугментаций

## Лицензия
MIT License

## Контакты

Василий

GitHub: https://github.com/svasly
Email: svasly2019@gmail.com