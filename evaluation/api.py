from typing import List

from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG
from dataloader.dataset_factory import _get_data
from evaluation.evaluation_db import EvaluationDatabase
from evaluation.report import generate_model_performance_report, generate_short_report

EVALUATION_DATABASE_SINGLETON = EvaluationDatabase()

def evaluate_predictions_on_test_set(dataset_name: str, predicition: List[int], data_set_root: str, _db=None, random_seed: int=0):
    if _db is None:
        _db = EVALUATION_DATABASE_SINGLETON
    _db.add_predictions(dataset_name=dataset_name, y=predicition, random_seed=random_seed)
    if not _db.has_groundtruths_for(dataset_name):
        _db.add_groundtruths(dataset_name=dataset_name, y=_get_ground_truths(dataset_name, data_set_root=data_set_root))


def report_model_performance(for_datasets=VALID_DATASET_CONFIG, _db=None, short=False, verbose=True):
    if _db is None:
        _db = EVALUATION_DATABASE_SINGLETON

    if short:
        report = generate_short_report(for_datasets=for_datasets, db=_db)
    else:
        report = generate_model_performance_report(for_datasets=for_datasets, db=_db)
    
    if verbose:
        print(report)
    return report


def _get_ground_truths(dataset_name: str, data_set_root: str):
    dataset_info = _get_data(dataset_name, without_test_labels=False, data_set_root=data_set_root)
    if dataset_info is None:
        raise RuntimeError(f"Unknown data set: {dataset_name}")

    gts = [y for _, y, _ in dataset_info.datasets.test]
    return gts

def init_evaluation_database(verbose=True, hparams=None):
    global EVALUATION_DATABASE_SINGLETON
    EVALUATION_DATABASE_SINGLETON = EvaluationDatabase(hparams=hparams)
    if verbose:
        print("EvaluationDatabase cleared.")

if __name__ == '__main__':
    from evaluation.result_struct import parse_log

    hparams = {'model': 'test___', 'pca': False, 'components': None}
    EVALUATION_DATABASE_SINGLETON = EvaluationDatabase(hparams=hparams)
    init_evaluation_database(hparams=hparams)

    PATH = "/data2/datasets/deephs_benchmark"

    for config in VALID_DATASET_CONFIG:
        EVALUATION_DATABASE_SINGLETON.add_predictions(config, 0, [1, 0, 0])
        EVALUATION_DATABASE_SINGLETON.add_predictions(config, 2, [0, 1, 0])
        EVALUATION_DATABASE_SINGLETON.add_groundtruths(config, [0, 0, 0])
    report = report_model_performance(_db=EVALUATION_DATABASE_SINGLETON)
    
    o = parse_log(report)
    print(o)


