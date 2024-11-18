import os
import sys

from PIL import Image

from src import (
    Config,
    BaseConfig,
    CLI,
    Data,
)
from src import (
    Eyeriss,
)

import numpy as np

import matplotlib.pyplot as plt


def main(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    config = BaseConfig()

    print("Eyeriss configuration:")
    print(f"  Rows: {config.rows}")
    print(f"  Columns: {config.cols}")
    print(f"  Latency: {config.latency}")
    print(f"  Energy: {config.energy}")

    print("Starting emulator...")
    eyeriss = Eyeriss.from_config(config)

    print("\tSetting image and kernel...")
    eyeriss.set_filter(kernel)
    group_size = kernel.shape[0] + config.cols - 1
    # print("\tProcessing image in groups of rows up to group_size at a time...")
    row_groups = [image[i:i + group_size] for i in range(0, image.shape[0], group_size)]
    results = []

    for idx, group in enumerate(row_groups):
        print(f"\tProcessing group {idx + 1}/{len(row_groups)}...")
        eyeriss.set_image(group)
        result_rows = eyeriss()
        results.append(result_rows)

    result = np.vstack(results)

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file '{image_path}' not found")
        sys.exit(1)

    image = Image.open(image_path)
    image = image.resize((512, 512))
    image = np.array(image)
    image = image / 255.0 # normalize
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

    result = main(image, kernel)
    print("\n\tResult:", end=" ")
    print(result.shape)

    expected = np.zeros_like(image)
    for i in range(image.shape[0] - kernel.shape[0] + 1):
        for j in range(image.shape[1] - kernel.shape[1] + 1):
            expected[i, j] = np.sum(image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)

    source_image = Image.open(image_path)
    source_image = source_image.resize((512, 512))
    expected = expected * 255.0
    expected_image = Image.fromarray(expected.astype(np.uint8))
    expected_image.save("expected.png")
    result = result * 255.0
    result_image = Image.fromarray(result.astype(np.uint8))
    result_image.save("result.png")

    plt.subplot(1, 3, 1)
    plt.imshow(source_image, cmap="gray")
    plt.title("Source Image")

    plt.subplot(1, 3, 2)
    plt.imshow(expected, cmap="gray")
    plt.title("Expected Result")

    plt.subplot(1, 3, 3)
    plt.imshow(result_image, cmap="gray")
    plt.title("Actual Result")

    plt.show()
