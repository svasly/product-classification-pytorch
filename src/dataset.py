import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_transforms():
    """Создаем разные наборы преобразований для обучения и валидации/теста."""
    # Для обучения: добавляем случайности (аугментацию)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Случайное отражение по горизонтали
        transforms.RandomRotation(10),           # Случайный поворот на ±10 градусов
        transforms.ToTensor(),			 # Конвертируем в тензор
        transforms.Normalize((0.5,), (0.5,))	 # Нормализация
    ])

    # Для валидации и теста: только приведение к тензору и нормализация
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, test_transform

def get_dataloaders(batch_size=64, num_workers=2):
    train_transform, test_transform = get_transforms()

    # 1. Загружаем два объекта датасета (один с аугментацией, другой — чистый)
    # Они ссылаются на одни и те же данные на диске, так что память не дублируется
    full_train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
    full_val_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=test_transform)

    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

    # 2. Генерируем индексы для разделения (80% / 20%)
    dataset_size = len(full_train_ds)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42))

    train_size = int(0.8 * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 3. Создаем сабсеты (подмножества)
    # Тренировочный берет данные из "шумного" датасета, валидационный — из "чистого"
    train_dataset = Subset(full_train_ds, train_indices)
    val_dataset = Subset(full_val_ds, val_indices)

    # 4. Создаем DataLoader'ы
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Проверка
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    print(f"Загрузка завершена успешно!")
    print(f"Батчей в Train: {len(train_loader)} (всего ~{len(train_loader)*32} картинок)")
    print(f"Батчей в Val:   {len(val_loader)}")
    print(f"Батчей в Test:  {len(test_loader)}")

    # Проверка формы одного батча
    images, labels = next(iter(train_loader))
    print(f"Размер батча: {images.shape}") # [32, 1, 28, 28]
