import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression

from dataloader.dataset_factory import get_data
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG

from evaluation.api import evaluate_predictions_on_test_set, report_model_performance

import argparse

if __name__ == '__main__':
    start_ts = time.time()
    print('### PLS-DA Training + Eval. ###')

    parser = argparse.ArgumentParser()
    parser.add_argument("data_set_root")
    opt = parser.parse_args()

    for config in VALID_DATASET_CONFIG:
        print('\nDatset config. {}'.format(config))

        # get data
        data = get_data(config, data_set_root=opt.data_set_root)

        def get_x_y(dataset, labels=True):
            samples, features = len(dataset), len(dataset[0][2].wavelengths)
            xs, ys = np.zeros((samples, features)), np.zeros(samples)
            for idx, (x, y, _) in enumerate(dataset):
                x, y = np.array(x), np.array(y)
                xs[idx] = x[:, x.shape[1] // 2, x.shape[2] // 2]  # center pixel
                ys[idx] = y
            return xs, ys if labels else None

        encoder = OneHotEncoder(categories=[np.arange(0, len(data.info.classes))], sparse_output=False)

        x_train, y_train = get_x_y(data.datasets.train)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        # x_val, y_val = get_x_y(data.datasets.val)
        # y_val = encoder.fit_transform(y_val.reshape(-1, 1))
        # x_train, y_train = torch.cat((x_train, x_val)), torch.cat((y_train, y_val))
        x_test, _ = get_x_y(data.datasets.test, labels=False)

        # fit
        pls = PLSRegression(n_components=10, scale=False)
        pls.fit(x_train, y_train)

        # predict
        prediction = pls.predict(x_test).argmax(1)
        print('Prediction: {}'.format(prediction))

        evaluate_predictions_on_test_set(config, prediction, data_set_root=opt.data_set_root)

    report_model_performance()

    print(f"## Took {time.time() - start_ts} s")



