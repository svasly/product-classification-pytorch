import json
import matplotlib.pyplot as plt

def plot_training_history(file_path='history.json'):
    # 1. Загружаем данные из нашего JSON
    with open(file_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Создаем область для двух графиков
    plt.figure(figsize=(12, 5))

    # График Loss (Потери)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График Accuracy (Точность)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Сохраняем график как картинку
    plt.savefig('training_plots.png')
    print("📈 Графики сохранены в файл 'training_plots.png'")

    # Показываем окно с графиком (если есть графический интерфейс)
    plt.show()

if __name__ == '__main__':
    plot_history_path = 'history.json'
    try:
        plot_training_history(plot_history_path)
    except FileNotFoundError:
        print(f"❌ Файл {plot_history_path} не найден. Сначала запустите обучение!")
