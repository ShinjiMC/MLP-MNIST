import matplotlib.pyplot as plt
import re
import os

def cargar_mse_acc_por_epoch(path):
    epochs = []
    mses = []
    accs = []
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r"Epoch\s+(\d+)\s+-\s+MSE:\s+([0-9.eE+-]+)\s+-\s+ACC:\s+([0-9.eE+-]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                mses.append(float(match.group(2)))
                accs.append(float(match.group(3)))
    return epochs, mses, accs

logs = {
    "XOR": "./output/XOR/log.txt",
    "AND": "./output/AND/log.txt",
    "OR":  "./output/OR/log.txt"
}

colores = {
    "XOR": "red",
    "AND": "green",
    "OR": "orange"
}

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, path) in enumerate(logs.items()):
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        continue

    epochs, mses, accs = cargar_mse_acc_por_epoch(path)
    ax1 = axs[i]

    color_mse = colores[name]
    color_acc = "blue"

    ax1.plot(epochs, mses, color=color_mse, label="MSE", linewidth=2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE", color=color_mse)
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_yscale("log")
    ax1.set_title(f"{name}")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color=color_acc, label="Accuracy", linestyle="--", linewidth=2)
    ax2.set_ylabel("Accuracy", color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
