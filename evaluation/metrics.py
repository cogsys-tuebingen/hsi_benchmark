from typing import List, Tuple
import numpy as np
import scipy


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def calc_accuracy(pred: np.ndarray, gt: List[int]) -> Tuple[float, float | None]:
    samples, num_runs = pred.shape

    if num_runs == 1:
        match = _to_numpy(pred[:, 0]) == _to_numpy(gt)
        return np.average(match).item(), None
    else:
        match = _to_numpy(pred) == _to_numpy(gt)[:, None].repeat(num_runs, 1)
        accuracies = np.average(match, axis=0)
        return np.mean(accuracies).item(), np.std(accuracies).item()


# Binary metrics (per class):
def calc_precision(pred: np.ndarray, gt: List[int], class_idx) -> Tuple[float, float | None]:
    '''
    Precision = TP / (TP + FP)
    '''
    samples, num_runs = pred.shape

    if num_runs == 1:
        tp = np.logical_and(_to_numpy(pred[:, 0]) == class_idx, _to_numpy(gt) == class_idx).sum()
        fp = np.logical_and(_to_numpy(pred[:, 0]) == class_idx, _to_numpy(gt) != class_idx).sum()
        return 0.0 if (tp + fp == 0) else (tp / (tp + fp)), None
    else:
        tp = np.logical_and(_to_numpy(pred) == class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) == class_idx).sum(axis=0)
        fp = np.logical_and(_to_numpy(pred) == class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) != class_idx).sum(axis=0)
        precisions = [t / (t + f) if t + f > 0 else 0.0 for t, f in zip(tp, fp)]
        return np.mean(precisions).item(), np.std(precisions).item()


def calc_recall(pred: np.ndarray, gt: List[int], class_idx) -> Tuple[float, float | None]:
    '''
    Recall = TP / (TP + FN)
    '''
    samples, num_runs = pred.shape

    if num_runs == 1:
        tp = np.logical_and(_to_numpy(pred[:, 0]) == class_idx, _to_numpy(gt) == class_idx).sum()
        fn = np.logical_and(_to_numpy(pred[:, 0]) != class_idx, _to_numpy(gt) == class_idx).sum()
        return 0.0 if (tp + fn == 0) else (tp / (tp + fn)), None
    else:
        tp = np.logical_and(_to_numpy(pred) == class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) == class_idx).sum(axis=0)
        fn = np.logical_and(_to_numpy(pred) != class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) == class_idx).sum(axis=0)
        recalls = [t / (t + f) if t + f > 0 else 0.0 for t, f in zip(tp, fn)]
        return np.mean(recalls).item(), np.std(recalls).item()


def calc_f1(pred: np.ndarray, gt: List[int], class_idx) -> Tuple[float, float | None]:
    '''
    F1 score = harmonic mean of precision & recall
    '''
    samples, num_runs = pred.shape

    if num_runs == 1:
        precision = calc_precision(pred, gt, class_idx)[0]
        recall = calc_recall(pred, gt, class_idx)[0]
        if precision + recall == 0:
            return 0.0, None
        return 2 * (precision * recall) / (precision + recall), None
    else:
        tp = np.logical_and(_to_numpy(pred) == class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) == class_idx).sum(axis=0)
        fp = np.logical_and(_to_numpy(pred) == class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) != class_idx).sum(axis=0)
        fn = np.logical_and(_to_numpy(pred) != class_idx, _to_numpy(gt)[:, None].repeat(num_runs, 1) == class_idx).sum(axis=0)
        scores = (2 * tp) / (2 * tp + fp + fn)
        return np.mean(scores).item(), np.std(scores).item()


# Ordinal metrics (for fruit classification only):
def calc_mae(pred: np.ndarray, gt: List[int]) -> Tuple[float, float | None]:
    samples, num_runs = pred.shape

    if num_runs == 1:
        return np.average(np.abs(_to_numpy(pred[:, 0]) - _to_numpy(gt))).item(), None
    else:
        diff = np.abs(_to_numpy(pred) - _to_numpy(gt)[:, None].repeat(num_runs, 1))
        errors = np.average(diff, axis=0)
        return np.mean(errors).item(), np.std(errors).item()


def calc_pearson_corr(pred: np.ndarray, gt: List[int]) -> Tuple[float, float | None]:
    samples, num_runs = pred.shape

    if num_runs == 1:
        return scipy.stats.pearsonr(pred[:, 0], gt), None
    else:
        coeffs = np.apply_along_axis(lambda x: scipy.stats.pearsonr(x, gt).statistic if np.unique(x).size > 1 else 0, 0, pred)
        return np.mean(coeffs).item(), np.std(coeffs).item()


def calc_kendalls_tau(pred: np.ndarray, gt: List[int]) -> Tuple[float, float | None]:
    samples, num_runs = pred.shape

    if num_runs == 1:
        return scipy.stats.kendalltau(pred[:, 0], gt), None
    else:
        taus = np.apply_along_axis(lambda x: scipy.stats.kendalltau(x, gt).statistic if np.unique(x).size > 1 else 0, 0, pred)
        return np.mean(taus).item(), np.std(taus).item()
