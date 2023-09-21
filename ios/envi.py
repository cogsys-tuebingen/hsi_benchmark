import os
import numpy as np
import spectral.io.envi as envi
import spectral


def load_envi(path):
    _exts = '.bin'
    if not os.path.exists('%s%s' % (path, _exts)):
        _exts = '.img'
    if not os.path.exists('%s%s' % (path, _exts)):
        _exts = '.raw'

    if not os.path.exists('%s%s' % (path, _exts)):
        raise spectral.io.spyfile.FileNotFoundError("Could not find data for: %s" % path)

    envi_header = envi.open('%s.hdr' % path, image='%s%s' % (path, _exts))
    envi_data = envi_header.load()

    if envi_header.bands.centers is not None and len(envi_header.bands.centers) > 300:
        print(f"Apply binning to {path}")
        envi_data = simulate_binning(envi_data, 2)
    return envi_header, envi_data


def save_envi(path, _image, force=False):
    os.makedirs(path[:path.rfind("/")], exist_ok=True)

    envi.save_image('%s.hdr' % path, _image, ext='.bin', force=force)
    # print("Wrote envi file to: %s " % path)


def load_referenced_envi(path):
    _raw_envi_header, _raw_envi_data = load_envi(path)
    _white_envi_header, _white_envi_data = load_envi(path + "_White")
    _dark_envi_header, _dark_envi_data = load_envi(path + "_Dark")

    return _raw_envi_header, use_references(_raw_envi_data, _white_envi_data, _dark_envi_data)


def use_references(intensities, white_reference, dark_reference=None):
    if white_reference.ndim == 3:
        white_reference = white_reference.mean(axis=0)

    if dark_reference is None:
        return np.divide(intensities, white_reference, out=np.zeros_like(intensities), where=white_reference != 0)
    else:
        if dark_reference.ndim == 3:
            dark_reference = dark_reference.mean(axis=0)

        # there seems to be some bad referencing values
        # so set all negative values to zero.
        a = intensities - dark_reference
        # a[a < 0] = 0
        b = white_reference - dark_reference
        # b[b < 0] = 0
        return np.divide(a, b, out=np.zeros_like(intensities), where=b != 0)


def simulate_binning(x, bin_size):
    assert x.shape[2] % bin_size == 0
    out = np.zeros((x.shape[0], x.shape[1], x.shape[2] // bin_size), dtype=x.dtype)

    for bin in range(x.shape[2] // bin_size):
        out[:,:, bin] = np.average(x[:, :, [(bin_size*bin) + i for i in range(bin_size)]], axis=2)

    return out

