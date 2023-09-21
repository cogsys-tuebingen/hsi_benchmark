import glob
import os
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ios import envi


class HSDataset(Dataset, ABC):
    '''
    General structure: item (X), label (y), meta data
    '''
    def __init__(self, data_path: str, config: str, split: str = None, balance: bool = False, transform=None):
        self.data_path = data_path
        self.config = config

        self.split = split
        self.transform = transform

        self.records = self._get_records()

        self.balance = balance
        if self.balance:
            self._balance_classes()

    def __len__(self):
        return len(self.records)

    @abstractmethod
    def _get_records(self):
        pass

    def _balance_classes(self):
        max_count = 0
        for s in self.classes:
            count = len([r for r in self.records if r['class_label'] == s])
            print("%s #: %i" % (s, count))
            max_count = max(max_count, count)

        self.balance_to = max_count
        print("# Augment data to get balanced classes of size %i" %
              self.balance_to)

        target_class_size = self.balance_to
        # the sets are unbalanced, so augment the data to get balance sets
        for s in self.classes:
            class_records = [r for r in self.records if r['class_label'] == s]

            if len(class_records) == 0:
                continue

            print("# Augment: %s to %i elements" % (s, target_class_size))

            missing_objects_count = target_class_size - len(class_records)
            for i in range(missing_objects_count):
                new_record = class_records[np.random.randint(
                    0, len(class_records))]
                self.records = np.concatenate((self.records, [new_record]))

        print("# Data augmented")

        for s in self.classes:
            count = len([r for r in self.records if r['class_label'] == s])
            print("%s #: %i" % (s, count))

    @abstractmethod
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, object]:
        pass


def bands_as_first_dimension(_obj):
    if isinstance(_obj, torch.Tensor):
        return _obj.permute(2, 0, 1)
    else:
        return _obj.transpose((2, 0, 1))


def bands_as_first_dimension_rev(_obj):
    if isinstance(_obj, torch.Tensor):
        return _obj.permute(1, 2, 0)
    else:
        return _obj.transpose((1, 2, 0))


def find_hdr_files_in_path(data_path: str):
    list_of_files = glob.glob(os.path.join(data_path, "*.hdr"))
    if len(list_of_files) == 0:
        list_of_files = glob.glob(os.path.join(data_path, "**/*.hdr"))

    # remove the white and dark references
    main_files = list(
        filter(lambda x: not (x.endswith("_White.hdr") or x.endswith("_Dark.hdr")), list_of_files))

    # remove the data_path_prefix
    main_files = list(map(lambda x: x.replace(data_path + '/', ""), main_files))

    # remove the .hdr ending
    main_files = list(map(lambda x: x.replace(".hdr", ""), main_files))
    return main_files


def load_recording(path: str, spatial_size: Tuple):
    _header, _data = envi.load_envi(path)
    if spatial_size is not None:
        _data = cv2.resize(_data, dsize=spatial_size, interpolation=cv2.INTER_CUBIC)
    _data = np.array(_data)
    return bands_as_first_dimension(_data)


def resize_to_target_size(item: torch.Tensor, spatial_size: Tuple):
    if spatial_size is not None:
        item = bands_as_first_dimension_rev(item)
        item = cv2.resize(item.numpy(), dsize=spatial_size, interpolation=cv2.INTER_CUBIC)
        item = torch.from_numpy(bands_as_first_dimension(item))

    return item

def get_channel_wavelengths(meta_data) -> torch.Tensor:
    channel_wavelengths =  [d.wavelengths for d in meta_data]

    if isinstance(channel_wavelengths, list):
        if isinstance(channel_wavelengths[0], torch.Tensor):
            return torch.stack(channel_wavelengths)
        else:
            min_len = min([len(cw) for cw in channel_wavelengths])
            max_len = max([len(cw) for cw in channel_wavelengths])

            if min_len == max_len:
                t = torch.tensor(channel_wavelengths)
            else:
                t = torch.ones((len(channel_wavelengths), max_len))
                for i, cw in enumerate(channel_wavelengths):
                    t[i, :len(cw)] = torch.tensor(cw)

            if t.ndim < 2:
                return t.unsqueeze(1)
            else:
                return t
    else:
        return torch.tensor(channel_wavelengths).unsqueeze(1)


