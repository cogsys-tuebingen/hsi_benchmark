import torch
import math
import numpy as np
from torchvision.transforms import RandomResizedCrop, GaussianBlur


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def _random_flip(self, x: torch.Tensor):
        d = np.random.randint(0, 4)

        if d == 0:
            return x
        if d == 1:
            return x.flip(1)
        if d == 2:
            return x.flip(2)
        if d == 3:
            return x.flip([1, 2])

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_flip(x)
        return x, y, meta


class RandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p

    def _random_rotate(self, x):
        """
        rotates between -45° and 45°
        """
        random_degree = np.random.randint(-90, 90)

        theta = math.pi / 180 * random_degree
        rotation_matrix = torch.tensor(
            [[math.cos(theta), -math.sin(theta), 0],
             [math.sin(theta), math.cos(theta), 0],
             [0, 0, 1]], dtype=torch.float32).to(x.device)
        input_tf = self._affine2d(x,
                                  rotation_matrix,
                                  center=True,
                                  mode='nearest')
        return input_tf

    def _affine2d(self, x, matrix, mode='bilinear', center=True):
        """
        2D Affine image transform on torch.Tensor
        """
        if matrix.dim() == 2:
            matrix = matrix[:2, :]
            matrix = matrix.unsqueeze(0)
        elif matrix.dim() == 3:
            if matrix.size()[1:] == (3, 3):
                matrix = matrix[:, :2, :]

        A_batch = matrix[:, :, :2]
        if A_batch.size(0) != x.size(0):
            A_batch = A_batch.repeat(x.size(0), 1, 1)
        b_batch = matrix[:, :, 2].unsqueeze(1)

        # make a meshgrid of normal coordinates
        _coords = self._iterproduct(x.size(1), x.size(2))
        coords = _coords.unsqueeze(0).repeat(
            x.size(0), 1, 1).float().to(x.device)

        if center:
            # shift the coordinates so center is the origin
            coords[:, :, 0] = coords[:, :, 0] - (x.size(1) / 2. - 0.5)
            coords[:, :, 1] = coords[:, :, 1] - (x.size(2) / 2. - 0.5)
        # apply the coordinate transformation
        new_coords = coords.bmm(A_batch.transpose(
            1, 2)) + b_batch.expand_as(coords)

        if center:
            # shift the coordinates back so origin is origin
            new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(1) / 2. - 0.5)
            new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(2) / 2. - 0.5)

        # map new coordinates using bilinear interpolation
        if mode == 'nearest':
            x_transformed = self._nearest_interp2d(x.contiguous(), new_coords)
        elif mode == 'bilinear':
            x_transformed = self._bilinear_interp2d(x.contiguous(), new_coords)
        else:
            x_transformed = None

        return x_transformed

    def _nearest_interp2d(self, input, coords):
        """
        2d nearest neighbor interpolation th.Tensor
        """
        # take clamp of coords so they're in the image bounds
        x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 1).round()
        y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 1).round()

        stride = torch.LongTensor(input.stride())
        x_ix = x.mul(stride[1]).long()
        y_ix = y.mul(stride[2]).long()

        input_flat = input.view(input.size(0), -1)

        mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

        return mapped_vals.view_as(input)

    def _bilinear_interp2d(self, input, coords):
        """
        bilinear interpolation in 2d
        """
        x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 2)
        x0 = x.floor()
        x1 = x0 + 1
        y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 2)
        y0 = y.floor()
        y1 = y0 + 1

        stride = torch.LongTensor(input.stride())
        x0_ix = x0.mul(stride[1]).long()
        x1_ix = x1.mul(stride[1]).long()
        y0_ix = y0.mul(stride[2]).long()
        y1_ix = y1.mul(stride[2]).long()

        input_flat = input.view(input.size(0), -1)

        vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
        vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
        vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
        vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))

        xd = x - x0
        yd = y - y0
        xm = 1 - xd
        ym = 1 - yd

        x_mapped = (vals_00.mul(xm).mul(ym) +
                    vals_10.mul(xd).mul(ym) +
                    vals_01.mul(xm).mul(yd) +
                    vals_11.mul(xd).mul(yd))

        return x_mapped.view_as(input)

    def _iterproduct(self, *args):
        return torch.from_numpy(np.indices(args).reshape((len(args), -1)).T)

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_rotate(x)
        return x, y, meta


class RandomIntensityScale(object):
    def __init__(self, p=0.5, mean=1.0, std=0.5):
        self.p = p
        self.mean = mean
        self.std = std

    def _random_intensity_scale(self, x: torch.Tensor):
        scale = torch.normal(torch.tensor(self.mean), torch.tensor(self.std))
        return x * scale

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_intensity_scale(x)
        return x, y, meta


class RandomNoise(object):
    def __init__(self, p=0.5, factor=0.1):
        self.p = p
        self.factor = factor

    def _random_noise(self, x: torch.Tensor):
        _n = torch.normal(0, 1e-1, x.shape)
        return x + _n

    def _random_noise_v2(self, x: torch.Tensor):
        ch, h, w = x.shape

        _std = x.float().std((1, 2)).expand(h, w, ch).permute(2, 0, 1)
        _mean = x.float().mean((1, 2)).expand(h, w, ch).permute(2, 0, 1)

        _n = torch.normal(_mean, _std)
        return x + (self.factor * _n)

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_noise_v2(x)
        return x, y, meta


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def _random_resized_crop(self, x: torch.Tensor):
        input_size = x.shape[-1]
        return RandomResizedCrop(input_size, antialias=False)(x)

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_resized_crop(x)
        return x, y, meta


class RandomBlur(object):
    def __init__(self, p=0.5, kernel_size=7, sigma=1):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _random_gaussian_blur(self, x: torch.Tensor):
        return GaussianBlur(self.kernel_size, self.sigma)(x)

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_gaussian_blur(x)
        return x, y, meta


class RandomCut(object):
    def __init__(self, p=0.5,  max_ratio=0.5, num_cuts=2):
        self.p = p
        self.max_ratio = max_ratio
        self.num_cuts = num_cuts

    def _random_cut(self, x: torch.Tensor):
        h, w = x.shape[1:]

        x_ = x.clone()

        for i in range(self.num_cuts):
            ratio = np.random.rand(1) * (self.max_ratio / self.num_cuts)

            cut_w = int(w * ratio)
            cut_h = int(h * ratio)

            pos_x = np.random.randint(0 - cut_w, w)
            pos_y = np.random.randint(0 - cut_h, h)

            x_[:, max(0, pos_x):min(w, pos_x + cut_w), max(0, pos_y):min(h, pos_y + cut_h)] = 0.0

        return x_

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_cut(x)
        return x, y, meta


class RandomDropPixels(object):
    def __init__(self, p=0.5,  num_pixels=100):
        self.p = p
        self.num_pixels = num_pixels

    def _random_drop_pixels(self, x: torch.Tensor):
        mask_x = np.random.choice(range(x.shape[1]), self.num_pixels, replace=False)
        mask_y = np.random.choice(range(x.shape[2]), self.num_pixels, replace=False)

        x[:, mask_x, mask_y] = 0

        return x

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_drop_pixels(x)
        return x, y, meta


class RandomDropBands(object):
    def __init__(self, p=0.5, num_channels=20):
        self.p = p
        self.num_channels = num_channels

    def _random_drop_bands(self, x: torch.Tensor):
        all_channels = range(x.shape[0])
        mask_channels = np.random.choice(all_channels, self.num_channels, replace=False)

        x[mask_channels, :, :] = 0.0

        return x

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_drop_bands(x)
        return x, y, meta


class RandomDropCube(object):
    def __init__(self, p=0.5, width=10, height=10, num_channels=10):
        self.p = p
        self.width = width
        self.height = height
        self.num_channels = num_channels

    def _random_drop_cube(self, x: torch.Tensor):
        '''
        similar to: "Hyperspectral Image Classification Using Random Occlusion Data Augmentation"
        '''
        D, H, W = x.shape
        x_start = np.random.randint(W - self.width)
        y_start = np.random.randint(H - self.width)
        c_start = np.random.randint(D - self.num_channels)
        x_end = x_start + self.width
        y_end = y_start + self.height
        c_end = c_start + self.num_channels

        x[c_start:c_end, x_start:x_end, y_start:y_end] = 0.0

        return x

    def __call__(self, batch: torch.Tensor):
        x, y, meta = batch
        if np.random.rand(1) < self.p:
            x = self._random_drop_cube(x)
        return x, y, meta

