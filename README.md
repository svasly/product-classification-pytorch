# Product Classification with PyTorch

Классификация товаров по изображениям с использованием глубокого обучения.

## Описание

Проект для автоматической категоризации товаров маркетплейса на основе фотографий.

## Структура проекта

- `src/dataset.py` - загрузка и обработка данных
- `src/model.py` - архитектура нейросети
- `src/train.py` - процесс обучения
- `src/utils.py` - вспомогательные функции


## Требования

* **Python:** 3.12
* **CUDA Toolkit:** 12.4
* **PyTorch:** С поддержкой CUDA (рекомендуется установка через официальный сайт)

## Установка

**1. Создайте виртуальное окружение:**
```
   bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/macOS
   # или
   venv\Scripts\activate     # Для Windows
```

**2. Установите PyTorch с поддержкой CUDA:**

**2.1. Проверяем поддерживаемую версию:**

```
nvidia-smi
```

**2.2. Устанавливаем стабильную поддерживаемую свежую версию:**

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**2.3. Запукскаем проверку:**

```
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"
```

**3. Установите остальные зависимости:**

```
pip install -r requirements.txt
```

