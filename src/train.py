import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import json

# Импортируем наши модули
from dataset import get_dataloaders
from model import get_model


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Одна эпоха обучения.

    Теория:
    1. model.train() включает режим обучения (важно для Dropout и BatchNorm).
    2. optimizer.zero_grad(set_to_none=True) — более эффективный способ обнуления градиентов.
    3. loss.backward() вычисляет градиенты (обратное распространение ошибки).
    4. optimizer.step() обновляет веса модели.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        # Перенос данных на устройство (non_blocking=True ускоряет передачу для CUDA)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Обнуляем градиенты перед новым шагом
        optimizer.zero_grad(set_to_none=True)

        # Прямой проход (Forward pass)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход (Backward pass)
        loss.backward()

        # Gradient Clipping: ограничиваем норму градиентов значением 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Обновление весов
        optimizer.step()

        # Статистика (используем .detach() для исключения из графа вычислений)
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total, 100 * correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """
    Одна эпоха валидации.

    Теория:
    - model.eval() выключает Dropout и фиксирует статистику BatchNorm.
    - torch.no_grad() экономит память и время, не строя граф вычислений.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total, 100 * correct / total


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """
    Полный цикл обучения модели.
    """
    # Выбор устройства: CUDA (Nvidia), MPS (Apple Silicon) или CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model = model.to(device)

    # Функция потерь и современный оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Планировщик: снижает LR, если val_loss не падает 2 эпохи
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    best_epoch = 0          # FIX: инициализация переменной
    patience = 5            # FIX: параметр ранней остановки
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    try:
        for epoch in range(num_epochs):
            # Обучение и валидация
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            # Логирование
            for key, val in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc]):
                history[key].append(val)

            # Вывод метрик
            print(f"Epoch [{epoch+1:02d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            # Обновление LR
            scheduler.step(val_loss)

            # Сохранение лучшей модели (Checkpointing)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch  # FIX: обновляем метку последнего улучшения
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model.pth')
                print("New best model saved!")

            # FIX: Early Stopping — теперь переменные определены
            if epoch - best_epoch > patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # Загружаем лучшие веса перед возвратом
    if os.path.exists('best_model.pth'):
        # FIX: выровненные отступы в try/except
        try:
            checkpoint = torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=True
            )
        except Exception:
            # Если в чекпоинте есть сложные объекты, которые не проходят проверку
            checkpoint = torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=False
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')}")

    # Сохранение истории обучения
    with open('history.json', 'w') as f:
        json_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(json_history, f, indent=2)
    print("History saved to history.json")

    return model, history


if __name__ == '__main__':
    # Настройки для детерминированности (повторяемости) результатов
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Гиперпараметры
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3

    # Подготовка данных
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # Создание модели
    print("Initializing model...")
    model = get_model(num_classes=10, pretrained=True)

    # Запуск обучения
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )

    print("Training finished! Check best_model.pth and history.json")
