from typing import List, Tuple
import numpy as np


class EvaluationDatabase:
    def __init__(self, verbose= True, hparams=None):
        self.db_preds = {}
        self.db_gts = {}
        self.hparams = hparams

        self.verbose = verbose

    def add_predictions(self, dataset_name:str, random_seed: int, y: List[int]):
        self.db_preds.setdefault(dataset_name, {})
        if random_seed in self.db_preds[dataset_name].keys()and self.verbose:
            print(f"[EvaluationMetric] Overwrite prediction for {dataset_name} for random_seed {random_seed}")

        self.db_preds[dataset_name][random_seed] = y

    def has_groundtruths_for(self, dataset_name: str):
        return dataset_name in self.db_gts.keys()

    def add_groundtruths(self, dataset_name:str, y: List[int]):
        if dataset_name in self.db_gts.keys() and self.verbose:
            print(f"[EvaluationMetric] Overwrite ground truths for {dataset_name}")

        self.db_gts[dataset_name] = y

    def get_pair_for(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        cond = dataset_name in self.db_preds.keys() and dataset_name in self.db_gts.keys()

        if cond:
            pred_lens = [len(v) for _, v in self.db_preds[dataset_name].items()]
            if max(pred_lens) != min(pred_lens):
                raise RuntimeError(f"[EvaluationMetric] {dataset_name}: length of the predicitions do not match!")
            
            if pred_lens[0] != len(self.db_gts[dataset_name]):
                raise RuntimeError(f"[EvaluationMetric] {dataset_name}: length of ground truths and predicitions do not match!")

        return np.array([v for _,v in self.db_preds[dataset_name].items()]).transpose(1, 0), np.array(self.db_gts[dataset_name])

    def available_for(self, dataset_name: str):
        cond = dataset_name in self.db_preds.keys() and dataset_name in self.db_gts.keys()

        if cond:
            pred_lens = [len(v) for _, v in self.db_preds[dataset_name].items()]
            if max(pred_lens) != min(pred_lens):
                if self.verbose:
                    print(f"[EvaluationMetric] {dataset_name}: length of the predicitions do not match!")
                return False
            
            if pred_lens[0] != len(self.db_gts[dataset_name]):
                if self.verbose:
                    print(f"[EvaluationMetric] {dataset_name}: length of ground truths and predicitions do not match!")
                return False

        return cond

