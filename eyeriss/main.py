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


def main():
    config = BaseConfig()

    print("Eyeriss configuration:")
    print(f"  Rows: {config.rows}")
    print(f"  Columns: {config.cols}")
    print(f"  Latency: {config.latency}")
    print(f"  Energy: {config.energy}")

    print("Starting emulator...")
    # eyeriss.start()
    eyeriss = Eyeriss.from_config(config)

    image = np.random.rand(512, 512)
    kernel = np.random.rand(3, 3)
    plt.figure(0)
    plt.imshow(image, cmap='gray')

    print("\tSetting image and kernel...")
    eyeriss.set_filter(kernel)
    # eyeriss.set_image(image)
    # print("\tProcessing image in groups of rows up to 14 at a time...")
    row_groups = [image[i:i + 14] for i in range(0, image.shape[0], 14)]
    results = []

    for idx, group in enumerate(row_groups):
        print(f"\tProcessing group {idx + 1}/{len(row_groups)}...")
        eyeriss.set_image(group)
        result = eyeriss()
        results.append(result)

    result = np.vstack(results)

    # print("\tRunning inference...")
    # result = eyeriss()

    print("\tResult:")
    print(result.shape)
    plt.figure(1)
    plt.imshow(result, cmap='gray')
    plt.show()

    print("Closing emulator...")
    # eyeriss.close()

if __name__ == "__main__":
    main()