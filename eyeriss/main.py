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


def main():
    config = BaseConfig()
    eyeriss = Eyeriss.from_config(config)

    print("Eyeriss configuration:")
    print(f"  Rows: {config.rows}")
    print(f"  Columns: {config.cols}")
    print(f"  Latency: {config.latency}")
    print(f"  Energy: {config.energy}")

    print("Starting emulator...")
    eyeriss.start()

    image = np.random.rand(3, 512)
    kernel = np.random.rand(3, 3)

    print("\tSetting image and kernel...")
    eyeriss.set_filter(kernel)
    eyeriss.set_image(image)

    print("Closing emulator...")
    eyeriss.close()

if __name__ == "__main__":
    main()