import os
from collections import namedtuple
from typing import Tuple
import torch
import tqdm as tqdm

from dataloader.basic_dataloader import HSDataset, load_recording
from camera_definitions import CameraType, get_wavelengths_for
from dataloader.fruit.fruit_definitions import ClassificationType, Fruit, RipenessState, FirmnessLevel, SugarLevel
from dataloader.fruit.full_list import all_measurements
from dataloader.splits import fruit_sets


CLASSIFICATION_TYPE_2_CLASS_LABEL_2_IDX_MAPPING = {
    ClassificationType.RIPENESS: [RipenessState.UNRIPE, RipenessState.RIPE, RipenessState.OVERRIPE],
    ClassificationType.FIRMNESS: [FirmnessLevel.TOO_HARD, FirmnessLevel.READY, FirmnessLevel.TOO_SOFT],
    ClassificationType.SUGAR: [SugarLevel.NOT_SWEET, SugarLevel.READY, SugarLevel.TOO_SWEET]}

FruitMetaData = namedtuple('FruitMetaData', ['class_label', 'fruit', 'id', 'side', 'camera_type', 'wavelengths', 'filename', 'config'])


class FruitDataset(HSDataset):
    def __init__(self, data_path: str, config: str,
                 fruit: Fruit = Fruit.ALL, camera_type: CameraType = CameraType.ALL, classification_type: ClassificationType = ClassificationType.RIPENESS,
                 split: str = None, balance: bool = False, transform=None, target_size: Tuple = None):
        self.fruit = fruit
        self.camera_type = camera_type
        self.classification_type = classification_type
        self.target_size = target_size
        self.classes = [c.value for c in CLASSIFICATION_TYPE_2_CLASS_LABEL_2_IDX_MAPPING[classification_type]]

        super().__init__(data_path, config, split, balance, transform)

    def _get_records(self):
        _records = all_measurements

        if self.split is not None:
            assert self.split in ('train', 'val', 'test')
            train_records, val_records, test_records = split_train_val_test_set(_records)
            if self.split == 'train':
                _records = train_records
            elif self.split == 'val':
                _records = val_records
            elif self.split == 'test':
                _records = test_records

        records = []

        for r in tqdm.tqdm(_records):
            if self.fruit == Fruit.ALL or r.fruit == self.fruit:
                if self.camera_type == CameraType.ALL or r.camera_type == self.camera_type:
                    if r.is_labeled():
                        if self.classification_type == ClassificationType.RIPENESS:
                            class_label = merge_ripeness_levels(r.label.ripeness_state)
                        if self.classification_type == ClassificationType.FIRMNESS:
                            class_label = r.label.get_firmness_level()
                        if self.classification_type == ClassificationType.SUGAR:
                            if r.fruit == Fruit.AVOCADO:
                                continue
                            else:
                                class_label = r.label.get_sugar_level()

                        if class_label in [RipenessState.UNKNOWN, FirmnessLevel.UNKNOWN, SugarLevel.UNKNOWN]:
                            continue

                        records.append(
                            {
                                'path': os.path.join(self.data_path, r.get_file_path()),
                                'filename': r.get_file_path().split('/')[-1],
                                'class_label': class_label.value,
                                'class_id': self.classes.index(class_label.value),
                                'fruit': r.fruit,
                                'id': r.id,
                                'side': r.side,
                                'camera_type': r.camera_type,
                                'wavelengths': get_wavelengths_for(r.camera_type)
                            }
                        )

        return records

    def __getitem__(self, index):
        sample = self.records[index]
        label = torch.tensor(sample['class_id'])
        meta_data = FruitMetaData(
            class_label=sample['class_label'],
            fruit=sample['fruit'],
            id=sample['id'],
            side=sample['side'],
            camera_type=sample['camera_type'],
            wavelengths=sample['wavelengths'],
            filename=sample['filename'],
            config=self.config
        )

        item = torch.tensor(load_recording(sample['path'], spatial_size=self.target_size))

        if self.transform is not None:
            item, label, meta_data = self.transform([item, label, meta_data])

        return item, label, meta_data


def split_train_val_test_set(records):
    rest_set = []
    test_set = []
    for _l in records:
        if _l in fruit_sets.test_set:
            test_set.append(_l)
        else:
            rest_set.append(_l)

    train_set = []
    val_set = []
    for _l in rest_set:
        if _l in fruit_sets.validation_set:
            val_set.append(_l)
        else:
            train_set.append(_l)

    return train_set, val_set, test_set


def merge_ripeness_levels(state):
    if state == RipenessState.NEAR_OVERRIPE:
        state = RipenessState.OVERRIPE
    elif state == RipenessState.PERFECT:
        state = RipenessState.RIPE
    return state
