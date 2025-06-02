import os
from PIL import Image
import numpy as np

def convert_dataset(image_folder, labels_file, output_file):
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith('.png')
    ])

    if len(image_files) != len(labels):
        print(f"Error: {len(image_files)} imágenes y {len(labels)} etiquetas.")
        return

    with open(output_file, 'w') as out:
        out.write("784 10\n")
        for i, filename in enumerate(image_files):
            path = os.path.join(image_folder, filename)
            img = Image.open(path).convert('L').resize((28, 28))
            data = np.array(img).astype(np.float32).flatten() / 255.0
            label = labels[i]
            one_hot = [0] * 10
            one_hot[label] = 1
            values = ' '.join(f"{x:.5f}" for x in data)
            label_str = ' '.join(str(v) for v in one_hot)
            out.write(f"{values} {label_str}\n")

    print(f"Dataset guardado en '{output_file}' con {len(image_files)} ejemplos.")

if __name__ == "__main__":
    image_folder = "./test/mnist45/"
    labels_file = "./test/mnist45/labels.txt"
    output_file = "./test/mnist45.txt"

    convert_dataset(image_folder, labels_file, output_file)
