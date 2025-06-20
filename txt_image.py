import os
from PIL import Image
import numpy as np

def load_conv_output_to_images(txt_file, channels, height, output_root="img_conv"):
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    total_lines_per_image = channels * height
    total_images = len(lines) // total_lines_per_image
    assert len(lines) % total_lines_per_image == 0, "Number of lines in the file is not a multiple of (channels * height)"
    os.makedirs(output_root, exist_ok=True)
    for img_idx in range(total_images):
        start_line = img_idx * total_lines_per_image
        img_folder = os.path.join(output_root, f"img{img_idx}")
        os.makedirs(img_folder, exist_ok=True)
        channels_data = []
        for ch in range(channels):
            ch_start = start_line + ch * height
            ch_end = ch_start + height
            channel_lines = lines[ch_start:ch_end]
            arr = [list(map(float, line.split())) for line in channel_lines]
            img = np.array(arr)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_uint8 = (img_norm * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(img_folder, f"channel{ch}.png"))
            channels_data.append(img_norm)
        if channels == 3:
            merged = np.stack(channels_data, axis=-1)  # (H, W, 3)
            merged_uint8 = (merged * 255).astype(np.uint8)
            Image.fromarray(merged_uint8).save(os.path.join(img_folder, "merged_rgb.png"))
    print(f"Completed saving images from conv output to {output_root}.")

# Si el conv2d output fue de tamaño [2 x 28 x 28]:
load_conv_output_to_images("output_conv.txt", channels=2, height=28, output_root="output_img")
