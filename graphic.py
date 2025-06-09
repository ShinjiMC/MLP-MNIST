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
    "SGD": "./output/MNIST_SGD_001/log.txt"
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

# Gráfico TRAIN global
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

colors = {"SGD": "blue", "RMS": "red", "ADAM": "green"}

for name in global_train_losses:
    ax1.plot(global_epochs, global_train_losses[name], label=f"{name} Loss", linewidth=2, linestyle="-", color=colors[name])

for name in global_train_accs:
    ax2.plot(global_epochs, global_train_accs[name], label=f"{name} Accuracy", linewidth=2, linestyle="--", color=colors[name])

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Training Accuracy")
ax1.grid(True)
ax2.set_ylim(0, 1.05)

# Combinar ambas leyendas automáticamente y mostrarlas en el gráfico
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize="large")

plt.title("Comparación Global de Training Loss y Accuracy")
plt.tight_layout()
plt.show()



# Gráfico TEST global
plt.figure(figsize=(14, 8))
for name, accs in global_test_accs.items():
    plt.plot(global_epochs, accs, label=f"{name}", linewidth=2)

plt.title("Comparación Global de Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.ylim(0.95, 1.001)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


etiquetas = [
    "ADAM - 50 epochs",
    "ADAM - 40 epochs",
    "ADAM - 30 epochs",
    "ADAM - 20 epochs",
    "ADAM - 10 epochs",
    "RMS - 50 epochs",
    "RMS - 40 epochs",
    "RMS - 30 epochs",
    "RMS - 20 epochs",
    "RMS - 10 epochs",
    "SGD - 50 epochs",
    "SGD - 40 epochs",
    "SGD - 30 epochs",
    "SGD - 20 epochs",
    "SGD - 10 epochs",
]

test_acc_values = [0.35, 0.35, 0.35, 0.275, 0.35, 
                   0.225, 0.275, 0.25, 0.325, 0.275,
                   0.35, 0.30, 0.35, 0.375, 0.425]
def extraer_epochs(label):
    return int(label.split('-')[1].strip().split()[0])
data = {
    "ADAM": [],
    "RMS": [],
    "SGD": []
}
epochs_ordenados = sorted({extraer_epochs(e) for e in etiquetas if e.startswith("ADAM")}, reverse=True)

for optim in data.keys():
    filtered = [(extraer_epochs(e), val) for e, val in zip(etiquetas, test_acc_values) if e.startswith(optim)]
    filtered.sort(key=lambda x: x[0], reverse=True)
    data[optim] = [val for _, val in filtered]

plt.figure(figsize=(12, 6))

colores = {"ADAM": "green", "RMS": "red", "SGD": "blue"}

for optim, valores in data.items():
    plt.plot(epochs_ordenados, valores, marker='o', linestyle='-', color=colores[optim], label=optim)

plt.xticks(epochs_ordenados, [f"{e} epochs" for e in epochs_ordenados], fontsize=10)
plt.ylim(0, 1.0)
plt.ylabel("External Test Accuracy")
plt.xlabel("Epochs")
plt.title("Comparación de Test Accuracy según configuración de entrenamiento")
plt.grid(True)
plt.legend(title="Optimizador")
plt.tight_layout()
plt.show()
