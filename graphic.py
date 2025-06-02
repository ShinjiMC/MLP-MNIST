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

logs = {
    "MNIST_50": "./output/MNIST_50/log.txt",
    "MNIST_MINI_25": "./output/MNIST_MINI_25/log.txt"
}

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


etiquetas = [
    "MNIST - 50 epochs",
    "MNIST - 40 epochs",
    "MNIST - 30 epochs",
    "MNIST - 20 epochs",
    "MNIST - 10 epochs",
    "MNIST - 5 epochs",
    "MNIST_MINI - 25 epochs",
    "MNIST_MINI - 20 epochs",
    "MNIST_MINI - 10 epochs",
    "MNIST_MINI - 5 epochs",
    "MNIST_3_Layers - 10 epochs"
]

test_acc_values = [0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.35, 0.35, 0.35, 0.35, 0.275]

x_pos = list(range(len(etiquetas)))

plt.figure(figsize=(12, 6))
plt.plot(x_pos, test_acc_values, marker='o', linestyle='-', color='purple')
plt.xticks(x_pos, etiquetas, rotation=45, ha='right', fontsize=9)
plt.ylim(0, 1.0)
plt.ylabel("External Test Accuracy")
plt.title("Comparación de Test Accuracy según configuración de entrenamiento")
plt.grid(True)
plt.tight_layout()
plt.show()