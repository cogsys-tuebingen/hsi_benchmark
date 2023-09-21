import os
from typing import List

import torchvision.transforms
from collections import namedtuple

from torchvision.transforms import Compose

from camera_definitions import str2camera_type, get_wavelengths_for, CameraType
from dataloader.basic_dataloader import HSDataset
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG, VALID_FRUIT_DATASET_CONFIG, VALID_HRSS_DATASET_CONFIG, VALID_DEBRIS_DATASET_CONFIG

from dataloader.fruit_dataloader import FruitDataset
from dataloader.fruit.fruit_definitions import str2classification_type, str2fruit

from dataloader.debris_dataloader import DebrisDataset

from dataloader.hrss_dataloader import RemoteSensingDataset, SCENE_2_CAMERA_MAPPING, str2scene, Scene
from sample_transforms import Normalize, RemoveLabel

VALID_DATASETS = ['debris', 'fruit', 'remote_sensing', 'waste']

DatasetInfo = namedtuple('DatasetInfo', ['config', 'task_type', 'classes', 'channels', 'image_size'])
DatasetSplits = namedtuple('DatasetSplits', ['train', 'val', 'test'])
class DataObject(object):
    def __init__(self,
                 config,
                 train_dataset: HSDataset, val_dataset: HSDataset, test_dataset: HSDataset,
                 task_type,
                 classes, channels, image_size):
        self.datasets = DatasetSplits(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset)
        self.info = DatasetInfo(
            config=config,
            task_type=task_type,
            classes=classes,
            channels=channels,
            image_size=image_size
        )

        print('(Train set size: {}, val set size: {}, test set size: {})'.format(len(self.datasets.train),
                                                                               len(self.datasets.val),
                                                                               len(self.datasets.test)))


def get_data(dataset_config, data_set_root, augmentations: List = []):
    return _get_data(dataset_config, without_test_labels=True, augmentations=augmentations, data_set_root=data_set_root)


def _get_data(dataset_config, data_set_root, without_test_labels=True, augmentations: List = []):
    DATASET_2_DIRECTORY_MAPPING = {
        'debris': os.path.join(data_set_root, 'deephs_debris_resized/'),  # TODO: set paths?!
        'waste': os.path.join(data_set_root, 'iosb_waste/'),  # TODO: set paths?!
        'fruit': os.path.join(data_set_root, 'deephs_fruit_v4/'),
        'remote_sensing': os.path.join(data_set_root, 'hrss_dataset')
        }
    assert dataset_config in VALID_DATASET_CONFIG

    dataset = dataset_config.split('/')[0]
    dataset_path = DATASET_2_DIRECTORY_MAPPING[dataset]
    assert os.path.exists(dataset_path)

    test_data_transforms = torchvision.transforms.Compose(
        [Normalize(dataset), RemoveLabel()]) if without_test_labels else Normalize(dataset)

    if dataset == 'debris':
        print('Load debris dataset with config. {}'.format(dataset_config))

        camera_type = str2camera_type(dataset_config.split('/')[1])
        wavelengths = None if camera_type == CameraType.ALL else get_wavelengths_for(camera_type)

        task_type = dataset_config.split('/')[-1]
        patch_size = 63 if task_type == 'patchwise' else None
        target_size = (128, 128) if task_type == 'objectwise' else None
        image_size = (patch_size, patch_size) if task_type == 'patchwise' else target_size

        train_data = DebrisDataset(dataset_path,
                                   config=dataset_config,
                                   split='train',
                                   camera_type=camera_type,
                                   patch_size=patch_size, dilation=30,
                                   target_size=target_size,
                                   transform=Compose([Normalize(dataset)] + augmentations),
                                   # drop_background=False,
                                   )
        val_data = DebrisDataset(dataset_path,
                                 config=dataset_config,
                                 split='val',
                                 camera_type=camera_type,
                                 patch_size=patch_size, dilation=30,
                                 target_size=target_size,
                                 transform=Normalize(dataset),
                                 # drop_background=False,
                                 )
        test_data = DebrisDataset(dataset_path,
                                  config=dataset_config,
                                  split='test',
                                  camera_type=camera_type,
                                  patch_size=patch_size, dilation=1,  # dilation=1 for test set
                                  target_size=target_size,
                                  transform=test_data_transforms,
                                  # drop_background=False,
                                  )

        return DataObject(dataset_config,
                          train_dataset=train_data, val_dataset=val_data, test_dataset=test_data,
                          task_type=task_type,
                          classes=train_data.classes, channels=wavelengths, image_size=image_size)
    elif dataset == 'fruit':
        print('Load fruit dataset with config. {}'.format(dataset_config))

        fruit = str2fruit(dataset_config.split('/')[1])
        classification_type = str2classification_type(dataset_config.split('/')[2])
        camera_type = str2camera_type(dataset_config.split('/')[-1])
        wavelengths = None if camera_type == CameraType.ALL else get_wavelengths_for(camera_type)

        target_size = (128, 128)
        image_size = target_size

        train_data = FruitDataset(dataset_path,
                                  config=dataset_config,
                                  split='train',
                                  fruit=fruit, camera_type=camera_type, classification_type=classification_type,
                                  target_size=target_size,
                                  transform=Compose([Normalize(dataset)] + augmentations))
        val_data = FruitDataset(dataset_path,
                                config=dataset_config,
                                  split='val',
                                  fruit=fruit, camera_type=camera_type, classification_type=classification_type,
                                target_size=target_size,
                                  transform=Normalize(dataset))
        test_data = FruitDataset(dataset_path,
                                 config=dataset_config,
                                split='test',
                                fruit=fruit, camera_type=camera_type, classification_type=classification_type,
                                 target_size=target_size,
                                transform=test_data_transforms)

        return DataObject(dataset_config,
                          train_dataset=train_data, val_dataset=val_data, test_dataset=test_data,
                          task_type='objectwise',
                          classes=train_data.classes, channels=wavelengths, image_size=image_size)

    elif dataset == 'remote_sensing':
        print('Load remote sensing scene {}'.format(dataset_config))

        scene = str2scene(dataset_config.split('/')[1])
        train_ratio = float(dataset_config.split('/')[2])
        camera_type = SCENE_2_CAMERA_MAPPING[scene]
        wavelengths = get_wavelengths_for(camera_type)

        patch_size = 63
        image_size = (patch_size, patch_size)

        dilation = 1

        train_data = RemoteSensingDataset(dataset_path,
                                          config=dataset_config,
                                          scene=scene,
                                          split='train',
                                          patch_size=patch_size, dilation=dilation,
                                          transform=Compose([Normalize(dataset)] + augmentations),
                                          drop_invalid=True,
                                          train_ratio=train_ratio
                                         )
        val_data = RemoteSensingDataset(dataset_path,
                                        config=dataset_config,
                                        scene=scene,
                                        split='val',
                                        patch_size=patch_size, dilation=dilation,
                                        transform=Normalize(dataset),
                                        drop_invalid=True,
                                        train_ratio=train_ratio
                                       )
        test_data = RemoteSensingDataset(dataset_path,
                                         config=dataset_config,
                                         scene=scene,
                                         split='test',
                                         patch_size=patch_size, dilation=1,  # dilation=1 for test set
                                         transform=test_data_transforms,
                                         drop_invalid=True,
                                         train_ratio=train_ratio
                                        )

        return DataObject(dataset_config,
                          train_dataset=train_data, val_dataset=val_data, test_dataset=test_data,
                          task_type='patchwise',
                          classes=train_data.classes, channels=wavelengths, image_size=image_size)
    raise RuntimeError(f"Dataset config '{dataset_config}' is unkown")


if __name__ == '__main__':
    configs = VALID_DATASET_CONFIG
    configs = sorted(configs)
    for config in configs:
        do = get_data(config, augmentations=[], data_set_root='/data2/datasets/deephs_benchmark')
        print(f"# {config}")
        print(f"# {len(do.datasets.train.records)}")
        print(f"# {len(do.datasets.val.records)}")
        print(f"# {len(do.datasets.test.records)}")
        print("-----")

