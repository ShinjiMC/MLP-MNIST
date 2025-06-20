import os
from PIL import Image
import numpy as np

def images_to_txt(image_folder, output_folder="output", output_filename="input.txt"):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        for filename in sorted(os.listdir(image_folder)):
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                image_path = os.path.join(image_folder, filename)
                img = Image.open(image_path).convert('RGB')
                arr = np.array(img) / 255.0  # Normalizar a [0, 1]
                # R, G, B canales como líneas separadas
                for channel in range(3):  # fila 0 = R, 1 = G, 2 = B
                    flat = arr[:, :, channel].flatten()
                    line = ' '.join(f'{val:.6f}' for val in flat)
                    f.write(line + '\n')
    print(f"Output written to {output_path}")

# USE
images_to_txt("./database/Simpson")
