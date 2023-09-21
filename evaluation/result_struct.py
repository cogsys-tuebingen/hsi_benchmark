from typing import NamedTuple, Dict, Tuple, Optional
import pickle
import codecs
import bz2
import re
import glob

from camera_definitions import CameraType

import enum
class TaskType(enum.Enum):
    OBJECTWISE = 'objectwise'
    PATCHWISE = 'patchwise'

class Accuracy(NamedTuple):
    mean: float | None
    std: float | None

class MAE(NamedTuple):
    value: float | None
    std: float | None

class OverallAccuracyResults(NamedTuple):
    remote_sensing: Accuracy
    debris: Accuracy
    fruit: Accuracy
    overall: Accuracy

class HyperparameterConfiguration(NamedTuple):
    model: str
    pca: bool
    components: int | None

class RemoteSensingResult(NamedTuple):
    overall: Accuracy
    train_ratio: Dict[float, Accuracy]

from dataloader.hrss_dataloader import Scene
class RemoteSensingResults(NamedTuple):
    dataset_wise: Dict[Scene, RemoteSensingResult]

class DebrisResult(NamedTuple):
    accuracy: Accuracy 

class DebrisConfiguration(NamedTuple):
    camera: CameraType 
    task_type: TaskType

class DebrisResults(NamedTuple):
    camera_wise: Dict[DebrisConfiguration, DebrisResult]

from dataloader.fruit.fruit_definitions import Fruit, ClassificationType

class FruitResult(NamedTuple):
    accuracy: Accuracy
    mae: MAE

class FruitConfiguration(NamedTuple):
    camera: CameraType | None
    classification_type: ClassificationType
    fruit: Fruit

class FruitResults(NamedTuple):
    camera_wise: Dict[FruitConfiguration, FruitResult]

class RawPredictionConfiguration(NamedTuple):
    dataset_config: str

class RawPrediction(NamedTuple):
    prediction: Tuple[Tuple[int, ...]]
    ground_truth: Tuple[int, ...]

# Main struct

class ExperimentResult(NamedTuple):
    hyperparameters: HyperparameterConfiguration | None
    remote_sensing: RemoteSensingResults
    debris: DebrisResults
    fruit: FruitResults
    overall: OverallAccuracyResults
    raw_predictions: Dict[RawPredictionConfiguration, RawPrediction]

# Util methods
def str2accuracy(s: str) -> Accuracy:
    s = s.replace(" ", "").replace("%", "")
    if "±" in s:
        s_splits = s.split("±")
        return Accuracy(float(s_splits[0]), float(s_splits[1]))
    else:
        return Accuracy(float(s), None)

def hparams2struct(hparams: dict) -> HyperparameterConfiguration:
    if hparams is None:
        return None

    return HyperparameterConfiguration(
        model=hparams['model'],
        pca=hparams['pca'],
        components=hparams['components']
    )

ENCODE_START = "#### ENCODED RESULTS -- START ####\n"
ENCODE_END = "#### ENCODED RESULTS -- END ####"
def encode_results(obj: ExperimentResult) -> str:
    s = pickle.dumps(obj)
    s = bz2.compress(s)
    s = codecs.encode(s, 'base64').decode()
    
    s = ENCODE_START + s + ENCODE_END
    return s

def decode_results(s: str) -> ExperimentResult:
    s = s.replace(ENCODE_START, "").replace(ENCODE_END, "")
    s = codecs.decode(s.encode(), 'base64')
    try:
        s = bz2.decompress(s)
    except OSError:
        pass
    return pickle.loads(s)

def parse_log(s: str) -> Optional[ExperimentResult]:
    m = re.search(ENCODE_START + "(.*)" + ENCODE_END, s, re.MULTILINE | re.DOTALL)
    if m is None:
        return None
    r = decode_results(m.group(1))
    return r

def parse_file(path: str) -> Optional[ExperimentResult]:
    f = open(path, 'r')
    return parse_log(f.read())

if __name__ == '__main__':
    def get_old_hparams(path):
        from slurm.plots.parse_log import _get_hyperparameters
        f = open(path, 'r')
        content = ''.join(f.readlines())
        return _get_hyperparameters(content)

    def replace_pickle(path, new_p):
        with open(path, 'r') as f:
            content = f.read()
        content = re.sub(ENCODE_START + "(.*)" + ENCODE_END, new_p, content, flags=re.MULTILINE | re.DOTALL)
        with open(path, 'w') as f:
            f.write(content)

    file_paths = glob.glob("/cshome/share/lvarga/deephs_benchmark_logs_july/job.*.out")

    for path in file_paths:
        print(path)
        o = parse_file(path)
        if o is not None:
            print(o.hyperparameters)
            o = ExperimentResult(
                debris=o.debris,
                fruit=o.fruit,
                overall=o.overall,
                raw_predictions=o.raw_predictions,
                remote_sensing=o.remote_sensing,
                hyperparameters=get_old_hparams(path))

            replace_pickle(path, encode_results(o))
    #    s = encode_results(o)
     #   print(len(s))
      #  print(decode_results(s) == o)

