import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)
import numpy as np

# Импортируем наши модули
from dataset import get_dataloaders
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
        weights_only=True  # FIX 1: Безопасная загрузка (только state_dict)
    )
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    """
    Полная оценка модели на тестовом наборе.

    Returns:
        dict с метриками и предсказаниями
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # FIX 2: non_blocking=True для ускорения передачи на GPU
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Конвертируем в массивы
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Основные метрики
    accuracy = accuracy_score(y_true, y_pred)

    # Метрики по классам
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)

    # Текстовый отчёт
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    return {
        'accuracy': float(accuracy),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'labels': y_true.tolist()
    }


def print_summary(metrics):
    """Выводит сводку по метрикам в консоль."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")

    print(f"\n{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 45)

    for i, name in enumerate(CLASS_NAMES):
        p = metrics['precision_per_class'][i]
        r = metrics['recall_per_class'][i]
        f = metrics['f1_per_class'][i]
        print(f"{name:<15} {p:<10.3f} {r:<10.3f} {f:<10.3f}")

    print("-" * 45)

    # Найдём лучший и худший классы по F1
    f1_scores = metrics['f1_per_class']
    best_idx = np.argmax(f1_scores)
    worst_idx = np.argmin(f1_scores)

    print(f"\nBest class (F1): {CLASS_NAMES[best_idx]} ({f1_scores[best_idx]:.3f})")
    print(f"Worst class (F1): {CLASS_NAMES[worst_idx]} ({f1_scores[worst_idx]:.3f})")
    print("=" * 60 + "\n")


def save_results(metrics, output_path='evaluation_results.json'):
    """Сохраняет результаты в JSON файл."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Строит и сохраняет матрицу ошибок.

    FIX 3: Обработка отсутствия библиотек и графического интерфейса.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Неинтерактивный бэкенд для серверов без GUI
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Warning: Could not import visualization libraries: {e}")
        print("Skipping confusion matrix plot.")
        return

    try:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")
        print("Continuing without visualization.")


if __name__ == '__main__':
    # Настройки
    MODEL_PATH = 'best_model.pth'
    BATCH_SIZE = 64

    # Выбор устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка данных (только тестовый набор)
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Загрузка модели
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)

    # Оценка
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)

    # Вывод сводки
    print_summary(metrics)

    # Сохранение результатов
    save_results(metrics)

    # Построение матрицы ошибок
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, CLASS_NAMES)

    print("Evaluation complete!")
