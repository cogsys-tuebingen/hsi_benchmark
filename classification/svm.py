import numpy as np
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from dataloader.dataset_factory import get_data
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG, \
    VALID_HRSS_DATASET_CONFIG, VALID_FRUIT_DATASET_CONFIG, VALID_DEBRIS_DATASET_CONFIG

from evaluation.api import evaluate_predictions_on_test_set, report_model_performance

import argparse

if __name__ == '__main__':
    start_ts = time.time()
    print('### SVM Training + Eval. ###')

    parser = argparse.ArgumentParser()
    parser.add_argument("data_set_root")
    opt = parser.parse_args()

    for config in VALID_DATASET_CONFIG:
        print('\nDatset config. {}'.format(config))

        # get data
        data = get_data(config, data_set_root=opt.data_set_root)

        def get_x_y(dataset, labels=True):
            samples, features = len(dataset), len(dataset[0][2].wavelengths)
            xs, ys = np.empty((samples, features)), np.empty(samples)
            for idx, (x, y, _) in enumerate(dataset):
                x = np.array(x)
                xs[idx] = x[:, x.shape[1] // 2, x.shape[2] // 2]  # center pixel
                ys[idx] = y
            return xs, ys if labels else None

        x_train, y_train = get_x_y(data.datasets.train)
        # x_val, y_val = get_x_y(data.datasets.val)
        # x_train, y_train = torch.cat((x_train, x_val)), torch.cat((y_train, y_val))
        x_test, _ = get_x_y(data.datasets.test, labels=False)

        # PCA preprocessing
        pca = PCA(n_components=10)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # fit
        svm = SVC(kernel='rbf')
        svm.fit(x_train, y_train)

        # predict
        prediction = svm.predict(x_test).astype(int)
        print('Prediction: {}'.format(prediction))

        evaluate_predictions_on_test_set(config, prediction, data_set_root=opt.data_set_root)

    report_model_performance()

    print(f"## Took {time.time() - start_ts} s")



