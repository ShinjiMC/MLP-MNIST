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
    "ADAM": "./output/MNIST_ADAM_001/log.txt",
    "RMS": "./output/MNIST_RMS_001/log.txt",
    "SGD": "./output/MNIST_SGD_001/log.txt",
    "ADAM_DROPOUT_0.2": "./output/MNIST_ADAM_DROP_02/log.txt",
    "ADAM_DROPOUT_0.5": "./output/MNIST_ADAM_DROP_05/log.txt",
    "ADAM_L2_0.01": "./output/MNIST_ADAM_L2_001/log.txt",
    "ADAM_L2_0.001": "./output/MNIST_ADAM_L2_0001/log.txt",
    "ADAM_DROPOUT_0.5_L2_0.01": "./output/MNIST_ADAM_DROP_L2/log.txt",
    "ADAM_DROPOUT_0.2_L2_0.001": "./output/MNIST_ADAM_DROP_L2_ALT/log.txt",
}

global_epochs = None
global_test_accs = {}
global_train_accs = {}
global_train_losses = {}

for name, path in logs.items():
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        continue

    epochs, train_losses, train_accs, test_accs = cargar_train_test_logs(path)
    global_epochs = epochs
    global_test_accs[name] = test_accs
    global_train_accs[name] = train_accs
    global_train_losses[name] = train_losses

    # Crear figura con 2 filas y 1 columna
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Entrenamiento y Evaluación - {name}", fontsize=14)

    # Gráfico 1: Pérdida y Accuracy de entrenamiento
    axs[0].plot(epochs, train_losses, label="Loss", color="red", linewidth=2)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss", color="red")
    axs[0].tick_params(axis='y', labelcolor="red")
    axs[0].set_title("Training Loss/Accuracy")
    axs[0].set_ylim(0.0, 1.0)
    axs[0].grid(True)

    ax2 = axs[0].twinx()
    ax2.plot(epochs, train_accs, label="Train Accuracy", color="blue", linestyle="--", linewidth=2)
    ax2.set_ylabel("Accuracy", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_ylim(0, 1.05)

    # Leyenda combinada para ambos ejes
    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc="upper right")

    # Gráfico 2: Accuracy de entrenamiento y prueba
    min_acc = min(min(train_accs), min(test_accs))
    max_acc = max(max(train_accs), max(test_accs))
    margin = 0.02  # Margen adicional para el eje Y

    axs[1].plot(epochs, train_accs, label="Train Accuracy", color="blue", linestyle="--", linewidth=2)
    axs[1].plot(epochs, test_accs, label="Test Accuracy", color="green", linewidth=2)
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Train/Test Accuracy")
    axs[1].set_ylim(max(0, min_acc - margin), min(1, max_acc + margin))  # Ajuste dinámico de límites
    axs[1].grid(True)
    axs[1].legend(loc="lower right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


datos = {
    "ADAM": {
        50: 0.35,
        40: 0.35,
        30: 0.35,
        20: 0.275,
        10: 0.35
    },
    "RMS": {
        50: 0.225,
        40: 0.275,
        30: 0.25,
        20: 0.325,
        10: 0.275
    },
    "SGD": {
        50: 0.35,
        40: 0.30,
        30: 0.35,
        20: 0.375,
        10: 0.425
    },
    "ADAM_DROPOUT_0.2":{
        10: 0.325,
        13: 0.325,
        20: 0.3,
        25: 0.275
    },
    "ADAM_DROPOUT_0.5": {
        5: 0.35,
        10: 0.275,
        20: 0.275,
        25: 0.3
    },
    "ADAM_L2_0.01": {
        7: 0.175,
        10: 0.225,
        20: 0.275,
        25: 0.275
    },
    "ADAM_L2_0.001": {
        10: 0.275,
        14: 0.3,
        20: 0.325,
        25: 0.325
    },
    "ADAM_DROPOUT_0.5_L2_0.01": {
        2: 0.175,
        10: 0.15,
        20: 0.15,
        25: 0.25,        
    },
    "ADAM_DROPOUT_0.2_L2_0.001": {
        10: 0.3,
        20: 0.35,
        25: 0.3 #best model
    }
}

# Agrupación de configuraciones
grupos = {
    "Base": ["ADAM", "RMS", "SGD"],
    "Dropout": [k for k in datos if "DROPOUT" in k and "L2" not in k],
    "L2": [k for k in datos if "L2" in k and "DROPOUT" not in k],
    "Dropout + L2": [k for k in datos if "DROPOUT" in k and "L2" in k]
}

colores_base = {"ADAM": "green", "RMS": "red", "SGD": "blue"}

for grupo, claves in grupos.items():
    plt.figure(figsize=(10, 6))
    
    for clave in claves:
        epochs = sorted(datos[clave].keys())
        accs = [datos[clave][ep] for ep in epochs]
        color = colores_base.get(clave, None)
        plt.plot(epochs, accs, marker='o', linestyle='-', label=clave, color=color)

    plt.title(f"{grupo} - Test Accuracy por Epoch")
    plt.xticks(sorted({ep for k in claves for ep in datos[k]}), rotation=45)
    plt.ylim(0, 0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()