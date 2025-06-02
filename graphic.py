import matplotlib.pyplot as plt
import re
import os

def cargar_train_test_logs(path):
    epochs = []
    train_losses = []
    train_accs = []
    test_accs = []

    with open(path, 'r') as f:
        for line in f:
            match = re.match(
                r"Epoch\s+(\d+),\s+Train Loss:\s+([0-9.eE+-]+),\s+Train Acc:\s+([0-9.]+)%,\s+Train Time:.*?,\s+Test Acc:\s+([0-9.]+)%",
                line
            )
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)) / 100)
                test_accs.append(float(match.group(4)) / 100)
    return epochs, train_losses, train_accs, test_accs

# Diccionario de logs
logs = {
    "MNIST_50": "./output/MNIST_50/log.txt",
    "MNIST_5": "./output/MNIST_5/log.txt",
    "MNIST_MINI": "./output/MNIST_MINI/log.txt"
}

# Graficar individualmente cada experimento
for name, path in logs.items():
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        continue

    epochs, train_losses, train_accs, test_accs = cargar_train_test_logs(path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Entrenamiento y Evaluación - {name}")

    # Gráfico 1: Entrenamiento
    axs[0].plot(epochs, train_losses, label="Loss", color="red", linewidth=2)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss", color="red")
    axs[0].tick_params(axis='y', labelcolor="red")
    axs[0].set_title("Training Loss/Acc")
    axs[0].grid(True)

    ax2 = axs[0].twinx()
    ax2.plot(epochs, train_accs, label="Accuracy", color="blue", linestyle="--", linewidth=2)
    ax2.set_ylabel("Accuracy", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_ylim(0, 1.05)

    # Gráfico 2: Test Accuracy
    axs[1].plot(epochs, test_accs, label="Test Accuracy", color="green", linewidth=2)
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Test Accuracy")
    axs[1].set_title("Test Accuracy")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Comparación de sobreajuste y rendimiento externo
etiquetas = [
    "MNIST_50 - 50 epochs",
    "MNIST_50 - 40 epochs",
    "MNIST_50 - 30 epochs",
    "MNIST_50 - 20 epochs",
    "MNIST_50 - 10 epochs",
    "MNIST_5 - 5 epochs",
    "MNIST_MINI - 10 epochs"
]
epoch_values = [50, 40, 30, 20, 10, 5, 10]
test_acc_values = [0.32, 0.32, 0.32, 0.32, 0.32, 0.40, 0.42]

plt.figure(figsize=(10, 6))
plt.plot(epoch_values, test_acc_values, marker='o', linestyle='-', color='purple')
for i, label in enumerate(etiquetas):
    plt.text(epoch_values[i], test_acc_values[i] + 0.01, label, fontsize=8, ha='center')
plt.xlabel("Epochs")
plt.ylabel("External Test Accuracy")
plt.title("Comparación de Test Accuracy según tamaño y duración de entrenamiento")
plt.grid(True)
plt.ylim(0, 1.0)
plt.show()

