from scipy import io
import os
import time

import numpy as np
import torch
import sklearn.decomposition
import matplotlib.pyplot as plt

from dataloader.basic_dataloader import bands_as_first_dimension
from dataloader.dataset_factory import DataObject, get_data
from dataloader.hrss_dataloader import Scene
from sample_transforms import RandomFlip, RandomRotate, RandomNoise, RandomCut, RandomCrop


class PCA(object):
    def __init__(self, data: DataObject, n_components: int = 10):
        # X_train = []
        # for (x, _, _) in data.datasets.train:
        #     x = np.array(x)
        #     x = x[:, x.shape[1] // 2, x.shape[2] // 2]  # center pixel
        #     X_train.append(x)
        # X_train = np.stack(X_train, axis=0)
        dataset = data.datasets.train
        X_train = np.empty((len(dataset), len(dataset[0][2].wavelengths)))
        for idx, (x, _, _) in enumerate(dataset):
            x = np.array(x)
            X_train[idx] = x[:, x.shape[1] // 2, x.shape[2] // 2]  # center pixel

        # calculate PCA -> components on training data set
        self.pca = sklearn.decomposition.PCA(n_components=n_components)
        self.pca.fit(X_train)

    def pca_transform(self, item):
        c, h, w = item.shape

        item = item.numpy().reshape((c, -1)).transpose(1, 0)

        # PCA transform/project to first n components
        item = self.pca.transform(item)

        item = item.transpose(1, 0).reshape((-1, h, w))

        return torch.from_numpy(item).float()

    def __call__(self, batch):
        batch_ = []
        for item in batch:
            batch_.append(self.pca_transform(item))
        return torch.stack(batch_)


if __name__ == '__main__':
    d = get_data('fruit/papaya/ripeness/SPECIM_FX10', augmentations=[], data_set_root='/data/')
    # d = get_data('debris/SPECIM_FX10/objectwise', augmentations=[], data_set_root='/data/')
    # d = get_data('remote_sensing/salinas/0.3', augmentations=[], data_set_root='/data/')

    # image = io.loadmat('/data/hrss_dataset/Indian_pines_corrected.mat')
    # image = torch.from_numpy(image["indian_pines_corrected"].astype(float))
    # image = bands_as_first_dimension(image)

    start_ts = time.time()
    pca = PCA(data=d, n_components=3)
    print(f"## Took {time.time() - start_ts} s")

    # image_transformed = pca.pca_transform(image)
    # print(image.shape, image_transformed.shape)
    # plt.imshow(torch.squeeze(image.mean(0)))
    # plt.savefig('/tmp/image.png')
    # plt.imshow(torch.squeeze(image_transformed[0]))
    # plt.savefig('/tmp/image_pca_comp.png')
    #
    test_item = d.datasets.test[0][0]
    print(test_item.shape)
    plt.imshow(torch.squeeze(test_item.mean(0)))
    plt.savefig('/tmp/test.png')
    test_item_transformed = pca.pca_transform(test_item)
    print(test_item_transformed.shape)
    plt.imshow(torch.squeeze(test_item_transformed[0]))
    plt.savefig('/tmp/test_pca_comp.png')
