import argparse
import os
from glob import glob

import cv2
import numpy as np


def generate_mooney(image_path, output_dir, blur_strength=15, white_pixel_ratio=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Couldn't read image: {image_path}. Skipping.")
        return

    ksize = int(blur_strength)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(1, ksize)

    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    percentile = (1.0 - np.clip(white_pixel_ratio, 0.0, 1.0)) * 100
    threshold_value = np.percentile(blurred, percentile)

    _, mooney = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_mooney{ext}")

    cv2.imwrite(output_path, mooney)
    print(f"[Success] Saved Mooney image: {output_path}")


def batch_process(inputs, output_dir, blur_strength, white_pixel_ratio):
    all_files = []

    for path in inputs:
        if os.path.isdir(path):
            extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
            for ext in extensions:
                all_files.extend(glob(os.path.join(path, ext)))
                all_files.extend(glob(os.path.join(path, ext.upper())))
        else:
            all_files.extend(glob(path))

    all_files = list(set(all_files))
    if not all_files:
        print(f"Nothing found in {path}")
        return

    print(f"Total of {len(all_files)} images")
    for file_path in all_files:
        generate_mooney(file_path, output_dir, blur_strength, white_pixel_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mooney Faces Generator")
    parser.add_argument("inputs", nargs="*", default=["data"], help="Path to og images")
    parser.add_argument("--output", default="mooneys", help="Output folder")
    parser.add_argument("--blur", type=float, default=21.0, help="Blur strength")
    parser.add_argument("--white-ratio", type=float, default=0.45,
                        help="Proportion of white pixels from 0.0 to 1.0")

    args = parser.parse_args()
    batch_process(args.inputs, args.output, args.blur, args.white_ratio)
