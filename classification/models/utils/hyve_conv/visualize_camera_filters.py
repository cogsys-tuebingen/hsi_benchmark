from classification.model_factory import load_model
from classification.models.utils.hyve_conv.hyve_convolution import HyVEConv
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def gauss(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-np.power(x - mean, 2)
                                                      / (2 * variance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    opt = parser.parse_args()

    if not os.path.exists(opt.checkpoint_path):
        print("! file does not exists")
        exit(-1)

    model, _ = load_model(opt.checkpoint_path)

    if model is None:
        print("! file can not be loaded")
        exit(-1)

    hyve_conv = None

    for module in model.children():
        if isinstance(module, HyVEConv):
            hyve_conv = module
            break

    if hyve_conv is None:
        print("! HyVEConv Layer not found")
        exit(-1)

    if isinstance(hyve_conv, HyVEConv):
        means, variances = hyve_conv.get_gauss().scaled_params()
        means, variances = means.detach(), variances.detach()
        wavelength_range = hyve_conv.wavelength_range

        steps = 1000
        xs = [wavelength_range[0] + i * (wavelength_range[1] - wavelength_range[0]) / steps for i in range(steps)]

        plt.figure()
        for mean, variance in zip(means, variances):
            plt.plot(xs, [gauss(x, mean, variance) for x in xs])
        plt.title("Camera filters")
        plt.xlabel("Wavelengths [nm]")
        plt.tight_layout()
        plt.show()


    else:
        raise RuntimeError()


