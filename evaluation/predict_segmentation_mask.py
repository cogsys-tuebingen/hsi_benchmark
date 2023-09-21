import argparse
from classification.model_factory import load_model
import torchvision

import torchvision.transforms

from torchvision.transforms import Compose

from camera_definitions import str2camera_type, get_wavelengths_for, CameraType
from dataloader.valid_dataset_configs import VALID_DATASET_CONFIG

from dataloader.debris_dataloader import DebrisDataset

from dataloader.hrss_dataloader import RemoteSensingDataset, SCENE_2_CAMERA_MAPPING, str2scene, Scene
from sample_transforms import Normalize, RemoveLabel
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm


def get_data(dataset_config, data_set_root):
    DATASET_2_DIRECTORY_MAPPING = {
        'debris': os.path.join(data_set_root, 'deephs_debris_resized/'),  # TODO: set paths?!
        'fruit': os.path.join(data_set_root, 'deephs_fruit_v4/'),
        'remote_sensing': os.path.join(data_set_root, 'hrss_dataset')
        }
    assert dataset_config in VALID_DATASET_CONFIG

    dataset = dataset_config.split('/')[0]
    dataset_path = DATASET_2_DIRECTORY_MAPPING[dataset]
    assert os.path.exists(dataset_path)

    test_data_transforms = torchvision.transforms.Compose([Normalize(dataset), RemoveLabel()])

    if dataset == 'debris':
        print('Load debris dataset with config. {}'.format(dataset_config))

        camera_type = str2camera_type(dataset_config.split('/')[1])
        wavelengths = None if camera_type == CameraType.ALL else get_wavelengths_for(camera_type)

        task_type = dataset_config.split('/')[-1]
        patch_size = 63 if task_type == 'patchwise' else None
        target_size = (128, 128) if task_type == 'objectwise' else None
        image_size = patch_size if task_type == 'patchwise' else target_size

        entire_data = DebrisDataset(dataset_path,
                                   split='train',
                                   camera_type=camera_type,
                                   patch_size=patch_size, dilation=1,
                                   target_size=target_size,
                                   transform=Compose([Normalize(dataset)]),
                                   drop_background=False,
                                   )

        return entire_data

    elif dataset == 'fruit':
        raise RuntimeError("Object-wise task not supported")

    elif dataset == 'remote_sensing':
        print('Load remote sensing scene {}'.format(dataset_config))

        scene = str2scene(dataset_config.split('/')[1])
        train_ratio = float(dataset_config.split('/')[2])
        camera_type = SCENE_2_CAMERA_MAPPING[scene]
        wavelengths = get_wavelengths_for(camera_type)

        patch_size = 63
        image_size = patch_size

        dilation = 1

        entire_data = RemoteSensingDataset(dataset_path,
                                          scene=scene,
                                          split='train',
                                          patch_size=patch_size, dilation=dilation,
                                          transform=Compose([Normalize(dataset)]),
                                          drop_invalid=True,
                                          train_ratio=train_ratio
                                         )

        return entire_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("data_root", type=str)
    opt = parser.parse_args()
    
    model, dataset_config = load_model(opt.checkpoint_path)
    model = model.cuda().eval()

    if model is None:
        print("Invalid checkpoint_path.")
        exit(-1)

    entire_data = get_data(dataset_config, opt.data_root)
    color_palette = sns.color_palette(n_colors=50)
    print(len(color_palette))

    if isinstance(entire_data, RemoteSensingDataset):
        segmentation_mask = np.zeros((entire_data.image.shape[1], entire_data.image.shape[2], 3))
        
        for sample in tqdm.tqdm(entire_data):
            x, _, meta = sample

            pos_x, pos_y = meta.x, meta.y
            segmentation_mask[pos_x, pos_y] = color_palette[model(x.cuda().unsqueeze(0)).argmax().cpu().item()]




        plt.figure()
        plt.imshow(segmentation_mask)
        plt.show()
