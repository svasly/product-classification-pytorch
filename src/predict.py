import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageChops
from torchvision import transforms
import sys
import os

# Импортируем нашу модель
from model import get_model

# Классы FashionMNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_model(checkpoint_path, device, num_classes=10):
    """Загружает модель из чекпоинта."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_image(image_path, model, device):
    """
    Улучшенная предобработка для реальных фото (Adidas, Nike и т.д.)

    FashionMNIST обучен на белых объектах на ЧЁРНОМ фоне.
    Реальные фото обычно имеют обратную схему, поэтому:
    1. Инвертируем цвета
    2. Обрезаем лишний фон
    3. Добавляем поля
    """
    # 1. Открываем и переводим в ч/б
    image = Image.open(image_path).convert('L')

    # 2. Инвертируем цвета (FashionMNIST: светлый объект на ЧЕРНОМ фоне)
    # Почти все фото из интернета имеют светлый фон, поэтому инверсия обязательна
    image = ImageOps.invert(image)

    # 3. Авто-обрезка лишнего фона (Crop)
    # Находим границы объекта и обрезаем пустоту вокруг
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)

    # 4. Добавляем небольшие поля (Padding), чтобы объект не касался краев
    width, height = image.size
    new_size = max(width, height) + 20
    new_img = Image.new("L", (new_size, new_size), (0))
    new_img.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
    image = new_img

    # 5. Финальные трансформации под размер 28x28
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Сохраним результат обработки для проверки (в ту же папку)
    # Посмотрите на этот файл: объект должен быть БЕЛЫМ на ЧЕРНОМ фоне
    image.save("debug_processed.png") 

    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return (
        CLASS_NAMES[predicted.item()],
        confidence.item(),
        probabilities.cpu().numpy()[0]
    )


def print_predictions(class_name, confidence, all_probs):
    """Выводит результаты предсказания."""
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 3 predictions:")
    print("-" * 50)

    # Сортируем вероятности
    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)

    for i in sorted_indices[:3]:
        print(f"  {CLASS_NAMES[i]:<15} {all_probs[i]:.2%}")

    print("=" * 50 + "\n")


if __name__ == '__main__':
    # Проверка аргументов
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path> [model_path]")
        print("\nExample:")
        print("  python src/predict.py test_image.jpg")
        print("  python src/predict.py test_image.jpg best_model.pth")
        sys.exit(1)

    # Выбор устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Путь к модели
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pth'

    # Проверка существования модели
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first: python src/train.py")
        sys.exit(1)

    # Загрузка модели
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Предсказание для пользовательского изображения
    image_path = sys.argv[1]

    try:
        print(f"\nProcessing image: {image_path}")
        class_name, confidence, all_probs = predict_image(image_path, model, device)
        print_predictions(class_name, confidence, all_probs)

        print("Debug: Processed image saved to 'debug_processed.png'")
        print("Check this file to verify preprocessing (should be WHITE object on BLACK background)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
