import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=10, pretrained=True):
    """
    Создаем адаптированную ResNet18 для FashionMNIST (28x28, 1 канал).

    Изменения по сравнению со стандартом:
    1. Конвертация первого слоя под 1 канал с суммированием весов (если pretrained=True).
    2. Уменьшение stride до 1 в первом слое, чтобы не терять детали на 28x28.
    3. Отключение MaxPool для сохранения пространственного разрешения.
    """
    # Загружаем ResNet18 с актуальным способом передачи весов
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Сохраняем старый слой для переноса знаний (RGB -> Grayscale)
    old_conv = model.conv1

    # Меняем слой: 1 канал, stride=1 (вместо 2), чтобы не уменьшать картинку 28x28 слишком рано
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

    if pretrained:
        with torch.no_grad():
            # Суммируем веса по 3 каналам в 1, чтобы сохранить "знания" о границах и формах
            model.conv1.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))

    # Для входного размера 28x28 стандартный MaxPool (2x2) слишком агрессивен.
    # Заменяем его на Identity (пустой слой), чтобы сохранить данные для глубоких слоев.
    model.maxpool = nn.Identity()

    # Заменяем классификатор под нужное количество классов
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == '__main__':
    # Автоматический выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация модели
    model = get_model(num_classes=10, pretrained=True).to(device)

    # Тестовый батч: 4 ч/б изображения 28x28 (стандарт FashionMNIST)
    x = torch.randn(4, 1, 28, 28).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Ожидаем [4, 10]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
