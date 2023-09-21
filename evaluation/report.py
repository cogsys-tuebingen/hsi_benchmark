from typing import Tuple
import tqdm
import tabulate
import numpy as np
from camera_definitions import CameraType

from evaluation.metrics import calc_accuracy, calc_mae, calc_pearson_corr, calc_kendalls_tau, calc_precision, \
    calc_recall, calc_f1
from evaluation.evaluation_db import EvaluationDatabase
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG
from evaluation.result_struct import Accuracy, MAE, DebrisConfiguration, ExperimentResult, FruitResults, OverallAccuracyResults, RawPrediction, RemoteSensingResults, decode_results, encode_results, hparams2struct, RawPredictionConfiguration, RawPrediction, encode_results , decode_results


def value_to_string(mean, std):
    if mean is None:
        return ""
    if std is None:
        return '{:.2f}'.format(mean)
    else:
        return '{:.2f} \261 {:.2f}'.format(mean, std)

def accuracy_to_string(mean, std):
    def _to_percentage(x):
        return "{:.2f}".format(100 * x)

    if mean is None:
        return ""
    if std is None:
        return f"{_to_percentage(mean)} %"
    else:
        return f"{_to_percentage(mean)} \261 {_to_percentage(std)} %"

def precision_recall_to_string(precision, recall):
    return '{:.3f} / {:.3f}'.format(precision, recall)


def _add_to_report(s: str, report):
    report += f"{s}\n"
    return report


def generate_model_performance_report(for_datasets, db: EvaluationDatabase):
    report = ""
        
    missing_datasets = []
    report = _add_to_report("###  Model Performance ###", report=report)
    report = _add_to_report("", report=report)

    complete_datasets = []
    raw_predictions = {}
    for dataset_name in for_datasets:
        if not db.available_for(dataset_name):
            missing_datasets.append(dataset_name)
            continue
        complete_datasets.append(dataset_name)

    it = tqdm.tqdm(complete_datasets)
    for dataset_name in it:
        it.set_description(f"Evaluate data set: {dataset_name}")

        report = _add_to_report(f"# Configuration {dataset_name}", report=report)
        preds, gt = db.get_pair_for(dataset_name)
        report = _add_to_report(f"-> Test accuracy  {accuracy_to_string(*calc_accuracy(preds, gt))}", report=report)
        raw_predictions[RawPredictionConfiguration(dataset_name)] = RawPrediction(tuple([tuple(p) for p in preds]), tuple(gt))

    fruit_report, fruit_results = _generate_fruit_report(db)
    debris_report, debris_results = _generate_debris_report(db)
    hrss_report, hrss_results = _generate_hrss_report(db)
    #report = _add_to_report(_generate_waste_report(db), report=report)

    if len(missing_datasets) > 0:
        report = _add_to_report(f"! Missing data sets [{len(missing_datasets)}]: {missing_datasets}", report=report)
    else:
        report = _add_to_report("# Evaluated all datasets.", report=report)

    overall_report, overall_results = _generate_overall_report(db)

    results = ExperimentResult(
        remote_sensing=hrss_results,
        debris=debris_results,
        fruit=fruit_results,
        overall=overall_results,
        hyperparameters=hparams2struct(db.hparams) if db.hparams is not None else None,
        raw_predictions=raw_predictions
    )
    encoded_results = encode_results(results)
    report = _add_to_report(encoded_results, report=report)
    s= decode_results(encoded_results)

    report = _add_to_report(fruit_report, report=report)
    report = _add_to_report(debris_report, report=report)
    report = _add_to_report(hrss_report, report=report)
    report = _add_to_report("", report=report)
    report = _add_to_report(overall_report, report=report)
    report = _add_to_report("", report=report)
    report = _add_to_report(f"### Done.", report=report)



    return report


def generate_short_report(for_datasets, db: EvaluationDatabase):
    report = ""

    complete_datasets = []
    for dataset_name in for_datasets:
        if not db.available_for(dataset_name):
            continue
        complete_datasets.append(dataset_name)

    for dataset_name in complete_datasets:
        report = _add_to_report(f"# Configuration {dataset_name}", report=report)
        report = _add_to_report(
            f"-> Test accuracy  {accuracy_to_string(*calc_accuracy(*db.get_pair_for(dataset_name)))}", report=report)

    return report


def _generate_fruit_report(db):
    from dataloader.fruit.fruit_definitions import Fruit, ClassificationType
    from evaluation.result_struct import FruitResult, FruitConfiguration
    report = "\n## Detailed report fruit data set:"

    classification_type = [f.value.lower() for f in ClassificationType]
    camera_types = [CameraType.SPECIM_FX10, CameraType.CORNING_HSI, CameraType.INNOSPEC_REDEYE]

    report = _add_to_report(f"", report=report)
    results = {}

    total_acc = []
    total_mae = []

    for ct in classification_type:
        report = _add_to_report(f"# Classification Type {ct.capitalize()}", report=report)
        table = [[""] + [ct.value for ct in camera_types] + ["Average"]]
        for ft in Fruit:
            row = [ft.value.capitalize() + '\n']
            avg_camera_acc = []
            avg_camera_mae = []
            for camera in camera_types:
                db_name = f"fruit/{ft.value.lower()}/{ct}/{camera.value}"
                if db_name not in VALID_DATASET_CONFIG:
                    row.append("-\n")
                    continue
                if db.available_for(db_name):
                    _acc = calc_accuracy(*db.get_pair_for(db_name))
                    avg_camera_acc.append(_acc[0])
                    total_acc.append(_acc[0])
                    _mae = calc_mae(*db.get_pair_for(db_name))
                    avg_camera_mae.append(_mae[0])
                    total_mae.append(_mae[0])
                    results[FruitConfiguration(
                        camera=camera,
                        classification_type=ct,
                        fruit=ft)] = FruitResult(Accuracy(*_acc), MAE(*_mae))

                    row.append(accuracy_to_string(*_acc) + '\n[MAE ' + value_to_string(*_mae) + ']')
                else:
                    row.append("*\n")
            cam_acc = float(np.average(np.array(avg_camera_acc))) if avg_camera_acc else None
            cam_mae = float(np.average(np.array(avg_camera_mae))) if avg_camera_mae else None
            results[FruitConfiguration(
                camera=None,
                classification_type=ct,
                fruit=ft)] = FruitResult(Accuracy(cam_acc, None), MAE(cam_mae, None))
            row.append(f"{accuracy_to_string(cam_acc, None)}\n [MAE {value_to_string(cam_mae, None)}]")
            table.append(row)
        report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)

    report = _add_to_report("*: predicitions missing; -: invalid configuration", report)

    report = _add_to_report("", report)
    report = _add_to_report('# Total average accuracy: ' + accuracy_to_string(np.average(np.array(total_acc)), None), report)
    report = _add_to_report('# Total average mean absolute error: ' + value_to_string(np.average(np.array(total_mae)), None), report)

    return report, FruitResults(camera_wise=results)
def _generate_waste_report(db):
    # FIXME: remove?
    from dataloader.waste_dataloader import CLASS_LABEL_2_ID_MAPPING
    report = "\n## Detailed report for waste data set:"

    camera_types = ['SPECIM_FX17']
    task_types = ['objectwise', 'patchwise']
    task_types = ['patchwise']
    classes = CLASS_LABEL_2_ID_MAPPING

    report = _add_to_report(f"", report=report)

    total_acc = []

    table = [[""] + camera_types + ["Average"]]
    for task in task_types:
        row = [task.capitalize()]
        avg_camera_acc = []
        for camera in camera_types:
            db_name = f"waste/{camera}/{task}"
            assert db_name in VALID_DATASET_CONFIG
            if db.available_for(db_name):
                _acc = calc_accuracy(*db.get_pair_for(db_name))
                row.append(accuracy_to_string(*_acc))
                avg_camera_acc.append(_acc[0])
                total_acc.append(_acc[0])
            else:
                row.append("*")
        row.append(accuracy_to_string(np.average(np.array(avg_camera_acc)), None) if avg_camera_acc else '')
        table.append(row)
    report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)
    report = _add_to_report("*: predicitions missing", report)

    report = _add_to_report("", report)
    report = _add_to_report('# Total average accuracy: ' + accuracy_to_string(np.average(np.array(total_acc)), None), report)

    report = _add_to_report('', report)
    report = _add_to_report('# Class-wise evaluation [Precision / Recall]:', report=report)
    for task in task_types:
        report = _add_to_report(f"# Task {task.capitalize()}", report=report)
        table = [['', 'Class'] + camera_types]
        for i, c in enumerate(classes):
            row = [i, c.capitalize()]
            for camera in camera_types:
                db_name = f"waste/{camera}/{task}"
                assert db_name in VALID_DATASET_CONFIG
                if db.available_for(db_name):
                    db_pair = db.get_pair_for(db_name)
                    _prec, _rec = calc_precision(*db_pair, class_idx=i), calc_recall(*db_pair, class_idx=i)
                    row.append(precision_recall_to_string(_prec[0], _rec[0]))
                else:
                    row.append("*")
            table.append(row)
        report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)
    report = _add_to_report("*: predicitions missing", report)

    return report


def _generate_debris_report(db):
    from dataloader.debris_dataloader import CLASS_LABEL_2_ID_MAPPING
    from evaluation.result_struct import TaskType, DebrisConfiguration, DebrisResult, DebrisResults
    report = "\n## Detailed report for debris data set:"

    camera_types = [CameraType.SPECIM_FX10, CameraType.CORNING_HSI]
    classes = CLASS_LABEL_2_ID_MAPPING

    report = _add_to_report(f"", report=report)
    results = {}

    total_acc = []

    table = [[""] + camera_types + ["Average"]]
    for task in TaskType:
        row = [task.value.capitalize()]
        avg_camera_acc = []
        for camera in camera_types:
            db_name = f"debris/{camera.value}/{task.value}"
            assert db_name in VALID_DATASET_CONFIG
            if db.available_for(db_name):
                _acc = calc_accuracy(*db.get_pair_for(db_name))
                row.append(accuracy_to_string(*_acc))
                avg_camera_acc.append(_acc[0])
                total_acc.append(_acc[0])
                results[DebrisConfiguration(
                    camera=camera,
                    task_type=task)] = DebrisResult(Accuracy(*_acc))
            else:
                row.append("*")
        row.append(accuracy_to_string(np.average(np.array(avg_camera_acc)), None) if avg_camera_acc else '')
        table.append(row)
    report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)
    report = _add_to_report("*: predicitions missing", report)

    report = _add_to_report("", report)
    report = _add_to_report('# Total average accuracy: ' + accuracy_to_string(np.average(np.array(total_acc)), None), report)

    report = _add_to_report('', report)
    report = _add_to_report('# Class-wise evaluation [Precision / Recall]:', report=report)
    for task in TaskType:
        report = _add_to_report(f"# Task {task.value.capitalize()}", report=report)
        table = [['', 'Class'] + [c.value for c in camera_types]]
        for i, c in enumerate(classes):
            row = [i, c.capitalize()]
            for camera in camera_types:
                db_name = f"debris/{camera.value}/{task.value}"
                assert db_name in VALID_DATASET_CONFIG
                if db.available_for(db_name):
                    db_pair = db.get_pair_for(db_name)
                    _prec, _rec = calc_precision(*db_pair, class_idx=i), calc_recall(*db_pair, class_idx=i)
                    row.append(precision_recall_to_string(_prec[0], _rec[0]))
                else:
                    row.append("*")
            table.append(row)
        report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)
    report = _add_to_report("*: predicitions missing", report)

    return report, DebrisResults(results)

def _generate_hrss_report(db) -> Tuple[str, RemoteSensingResults]:
    from dataloader.hrss_dataloader import SCENE_2_CAMERA_MAPPING, SCENE_2_LABEL_2_ID_MAPPING, str2scene, Scene
    from evaluation.result_struct import RemoteSensingResults, RemoteSensingResult
    report = "\n## Detailed report for HRSS:"

    report = _add_to_report(f"", report=report)
    results = {}

    total_acc = []

    for s in Scene:
        row = [s.value.capitalize()]
        per_dataset = {}
        for tr in [0.05, 0.1, 0.3]:
            db_name = f"remote_sensing/{s.value}/{tr}"
            assert db_name in VALID_DATASET_CONFIG
            if db.available_for(db_name):
                db_pair = db.get_pair_for(db_name)
                per_dataset[tr] = Accuracy(*calc_accuracy(*db_pair))


        results[s] = RemoteSensingResult(
            overall=per_dataset[0.3] if 0.3 in per_dataset.keys() else Accuracy(None, None),
            train_ratio=per_dataset)

    table = [['Scene'] + [""]]
    for s in Scene:
        row = [s.value.capitalize()]
        db_name = f"remote_sensing/{s.value}/0.3"
        assert db_name in VALID_DATASET_CONFIG
        if db.available_for(db_name):
            _acc = results[s].overall
            row.append(accuracy_to_string(_acc.mean, _acc.std))
            total_acc.append(_acc.mean)
        else:
            row.append("*")
        table.append(row)
    report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)

    report = _add_to_report("*: predicitions missing", report)
    report = _add_to_report("", report)

    report = _add_to_report('# Total average accuracy: ' + accuracy_to_string(np.average(np.array(total_acc)), None), report)

    report = _add_to_report('', report)
    report = _add_to_report('# Class-wise evaluation:', report=report)
    for s in Scene:
        db_name = f"remote_sensing/{s.value}/0.3"
        assert db_name in VALID_DATASET_CONFIG
        report = _add_to_report(f"# Scene {s.value.capitalize()}", report=report)
        table = [['', 'Class', 'Precision / Recall']]
        classes = SCENE_2_LABEL_2_ID_MAPPING[s]
        if db.available_for(db_name):
            db_pair = db.get_pair_for(db_name)
            for i, c in enumerate(classes):
                row = [i, c.capitalize()]
                _prec, _rec = calc_precision(*db_pair, class_idx=i), calc_recall(*db_pair, class_idx=i)
                row.append(precision_recall_to_string(_prec[0], _rec[0]))
                table.append(row)
        else:
            for i, c in enumerate(classes):
                row = [i, c.capitalize(), '*']
                table.append(row)
        report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)

    report = _add_to_report('', report=report)
    report = _add_to_report('# Impact of training set size:', report=report)
    table = [['Scene', "0.05", "0.1", "0.3"]]
    for s in Scene:
        row = [s.value.capitalize()]
        for tr in [0.05, 0.1, 0.3]:
            db_name = f"remote_sensing/{s.value}/{tr}"
            assert db_name in VALID_DATASET_CONFIG
            if db.available_for(db_name):
                db_pair = db.get_pair_for(db_name)
                _acc = results[s].train_ratio[tr]
                row.append(accuracy_to_string(_acc.mean, _acc.std))
            else:
                row.append('*')
        table.append(row)
    report = _add_to_report(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'), report)

    return report, results


def _generate_overall_report(db):
    from dataloader.fruit.fruit_definitions import Fruit, ClassificationType
    report = "\n## Overall report ##\n"

    def calc_average_acc_for_fruit(db):
        fruit_types = [f.value.lower() for f in Fruit]
        camera_types = ['SPECIM_FX10', 'CORNING_HSI', 'INNOSPEC_REDEYE']
        classification_type = [f.value.lower() for f in ClassificationType]
        total_acc = []

        for ct in classification_type:
            for ft in fruit_types:
                for camera in camera_types:
                    db_name = f"fruit/{ft}/{ct}/{camera}"
                    if db_name not in VALID_DATASET_CONFIG:
                        continue
                    if db.available_for(db_name):
                        _acc = calc_accuracy(*db.get_pair_for(db_name))
                        total_acc.append(_acc[0])
        return np.average(total_acc)

    def calc_average_acc_for_debris(db):
        camera_types = ['SPECIM_FX10', 'CORNING_HSI']
        task_types = ['objectwise', 'patchwise']

        total_acc = []
        task = task_types[0]
        for camera in camera_types:
            db_name = f"debris/{camera}/{task}"
            if db.available_for(db_name):
                _acc = calc_accuracy(*db.get_pair_for(db_name))
                total_acc.append(_acc[0])
            else:
                total_acc.append(-1)
        objectwise_acc = np.average(total_acc)

        total_acc = []
        task = task_types[1]
        for camera in camera_types:
            db_name = f"debris/{camera}/{task}"
            if db.available_for(db_name):
                _acc = calc_accuracy(*db.get_pair_for(db_name))
                total_acc.append(_acc[0])
            else:
                total_acc.append(-1)
        patchwise_acc = np.average(total_acc)


        return (objectwise_acc + patchwise_acc) / 2

    def calc_average_acc_for_hrss(db):
        from dataloader.hrss_dataloader import Scene
        total_acc = []

        for s in Scene:
            db_name = f"remote_sensing/{s.value}/0.3"
            if db.available_for(db_name):
                _acc = calc_accuracy(*db.get_pair_for(db_name))
                total_acc.append(_acc[0])
            else:
                total_acc.append(-1)

        return np.average(total_acc)

    results = [calc_average_acc_for_debris(db), calc_average_acc_for_fruit(db), calc_average_acc_for_hrss(db)]
    report = _add_to_report(f" Debris:   {accuracy_to_string(results[0], None)}", report=report)
    report = _add_to_report(f" Fruit:   {accuracy_to_string(results[1], None)}", report=report)
    report = _add_to_report(f" Remote Sensing:   {accuracy_to_string(results[2], None)}", report=report)
    report = _add_to_report(f" =>   {accuracy_to_string(np.average(results), None)}", report=report)
    report = _add_to_report(f"##########", report=report)

    return report, OverallAccuracyResults(
        debris=Accuracy(float(results[0]), None),
        fruit=Accuracy(float(results[1]), None),
        remote_sensing=Accuracy(float(results[2]), None),
        overall=Accuracy(float(np.average(results)), None),
    )
