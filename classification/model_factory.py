from typing import OrderedDict

import numpy as np
import torch
import os
from classification.models.deephs_net import DeepHSNet
from classification.models.deephs_hybrid_net import DeepHSHybridNet
from classification.models.deephs_hyve_net import DeepHSNet_with_HyVEConv, Larger_DeepHSNet_with_HyVEConv
from classification.models.mlp import MLP
from classification.models.cnn import CNN1D, CNN2D, CNN2DSpatial, CNN2DSpectral, CNN3D
from classification.models.resnet_hyper import ResNetHyper18
from classification.models.rnn import SimpleRNN
from classification.models.spectralnet import SpectralNet
from classification.models.hybridsn import HybridSN
from classification.models.resnet import ResNet152, ResNet18
from classification.models.resnet_hyve import ResNetHyve152, ResNetHyve18
from classification.models.spectralformer import ViT
from classification.models.gabor_cnn import GaborCNN
from classification.models.emp_cnn import EMPCNN
from classification.models.sstn import SSNet_AEAE
from classification.models.hit import HiT
from classification.models.hsi_attention import HSIAttentionModel2, HSIAttentionModel3, HSIAttentionModel4
from classification.models.deephs_hyper_net import DeepHSHybridNet_with_HyveConv

from classification.models.multitask_models import Multihead_DeepHSNet_with_HyVEConv, Multihead_ResNet_with_HyVEConv


VALID_MODELS = {
    'deephs_net': DeepHSNet,
    'deephs_net_pca': DeepHSNet,
    'deephs_hybrid_net': DeepHSHybridNet,
    'deephs_hyve_net': DeepHSNet_with_HyVEConv,
    'deephs_hyper_net': DeepHSHybridNet_with_HyveConv,
    'deephs_hyve_net_larger': Larger_DeepHSNet_with_HyVEConv,
    'gabor_cnn': GaborCNN,
    'emp_cnn': EMPCNN,
    'mlp': MLP,
    'cnn_1d': CNN1D,
    'cnn_2d': CNN2D,
    'cnn_2d_nopca': CNN2D,
    'cnn_2d_spatial': CNN2DSpatial,
    'cnn_2d_spatial_mean': CNN2DSpatial,
    'cnn_2d_spectral': CNN2DSpectral,
    'cnn_3d': CNN3D,
    'cnn_3d_nopca': CNN3D,
    'rnn': SimpleRNN,
    'resnet': ResNet18,
    'resnet152': ResNet152,
    'resnet_hyve': ResNetHyve18,
    'resnet152_hyve': ResNetHyve152,
    'resnet_hyper': ResNetHyper18,
    'hybridsn': HybridSN,
    'spectralnet': SpectralNet,
    'spectralformer': ViT,
    'sstn': SSNet_AEAE,
    'hit': HiT,
    'hsi_attention_2': HSIAttentionModel2,
    'hsi_attention_3': HSIAttentionModel3,
    'hsi_attention_4': HSIAttentionModel4,
}

VALID_MULTITASK_MODELS = {
    'deephs_hyve_net': Multihead_DeepHSNet_with_HyVEConv,
    'resnet_hyve': Multihead_ResNet_with_HyVEConv
}

HYVE_NETS = ('deephs_hyve_net', 'deephs_hyve_net_larger', 'resnet_hyve', 'resnet152_hyve', 'deephs_hyper_net', 'resnet_hyper')

VALID_MODELS_PRETRAINING = {'deephs_net', 'deephs_hyve_net', 'deephs_hyve_net_larger', 'resnet', 'resnet152', 'resnet_hyve', 'resnet152_hyve', 'deephs_hyper_net', 'resnet_hyper'}

DEFAULT_HPARAMS = {
    'deephs_net':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE','optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'deephs_net_pca':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE','optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 40},
    'deephs_hyve_net':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE','optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'deephs_hyve_net_larger':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE','optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'deephs_hybrid_net':
        {'batch_size': 4, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'deephs_hyper_net':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE','optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'mlp':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'cnn_1d':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'cnn_2d':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 40},
    'cnn_2d_nopca':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'cnn_2d_spatial':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 1},
    'cnn_2d_spatial_mean':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'cnn_2d_spectral':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'cnn_3d':
        {'batch_size': 8, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 40},
    'cnn_3d_nopca':
        {'batch_size': 1, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'gabor_cnn':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 3},
    'emp_cnn':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 3},
    'rnn':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'resnet':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'resnet_hyve':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'resnet_hyper':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'resnet152':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'resnet152_hyve':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'hybridsn':
        {'batch_size': 16, 'epochs': 50, 'lr': 0.0001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': True, 'components': 30},
    'spectralnet':
        {'batch_size': 8, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'SGD', 'scheduler': 'step', 'pca': False},
    'spectralformer':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'sstn':
        {'batch_size': 2, 'epochs': 50, 'lr': 0.01, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'hit':
        {'batch_size': 16, 'epochs': 100, 'lr': 0.001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'hsi_attention_2':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.0001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'hsi_attention_3':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.0001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
    'hsi_attention_4':
        {'batch_size': 32, 'epochs': 50, 'lr': 0.0001, 'loss': 'CE', 'optimizer': 'Adam', 'scheduler': 'step', 'pca': False},
}


def _add_additional_parameters(model, wavelengths, spatial_size):
    additional_args = {}
    if model in HYVE_NETS:
        additional_args['wavelength_range'] = (min(wavelengths), max(wavelengths))
        additional_args['num_of_wrois'] = 5
    if model in ('cnn_2d', 'cnn_2d_nopca', 'cnn_2d_spatial', 'cnn_2d_spatial_mean', 'cnn_2d_spectral',
                 'cnn_3d', 'hybridsn', 'spectralformer', 'gabor_cnn', 'emp_cnn'):
        additional_args['image_size'] = spatial_size
    if model.__contains__('cnn_2d_spatial'):
        additional_args['mode'] = 'mean' if model == 'cnn_2d_spatial_mean' else 'pca'

    return additional_args



def get_model(model, num_channels, num_classes, wavelengths, spatial_size=63, multitask=False) -> torch.nn.Module:
    if multitask:
        print('Getting multi-task classifier model {} ...'.format(model))
        assert model in VALID_MULTITASK_MODELS.keys()
        model = VALID_MULTITASK_MODELS[model](num_channels=num_channels, num_classes=num_classes, **_add_additional_parameters(model, wavelengths, spatial_size))
    else:
        print('Getting classifier model {}...'.format(model))
        assert model in VALID_MODELS.keys()
        model = VALID_MODELS[model](num_channels=num_channels, num_classes=num_classes, **_add_additional_parameters(model, wavelengths, spatial_size))
    return model


def get_pretrained_model(model,
                         path,
                         num_channels, num_classes, wavelengths, spatial_size=63,
                         reset_seed=0, reset_BN=False) -> torch.nn.Module:

    print('Load pretrained model ...')

    checkpoint = torch.load(path)
    assert model == checkpoint['model']
    model = get_model(
        model=checkpoint['model'],
        num_channels=checkpoint['num_channels'],
        num_classes=checkpoint['num_classes'],
        wavelengths=checkpoint['wavelengths'],
        spatial_size=checkpoint['spatial_size']
    )
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'],
                                                strict=False) # ignore task-specific model heads after potential multi-task pretraining
    if len(unexpected) > 0:
        print('Detected unexpected model params: {}'.format(unexpected))
    if len(missing) > 0:
        print('Detected missing model params: {}'.format(missing))

    # Reset model's first layer (for diff. wavelengths/channels)?
    if wavelengths is not None and wavelengths != checkpoint['wavelengths']:
        if checkpoint['model'] in HYVE_NETS:
            model.reset_first_layer(wavelength_range=(min(wavelengths), max(wavelengths)))
        else:
            model.reset_first_layer(num_channels=num_channels)
    # Reset model's head
    model.reset_head(seed=reset_seed, num_classes=num_classes, BN=reset_BN)

    return model

def get_default_model_hparams(model):
    hparams = DEFAULT_HPARAMS[model]
    return hparams

def save_model(model:str, dataset_config: str, num_channels: int, num_classes: int, wavelengths: list, spatial_size: int,
               model_state_dict: OrderedDict, output_path: str, name_addition: str=""):
    path = os.path.join(output_path, f"{model}_{dataset_config}_{name_addition}.pt".replace("/", "_").replace("\\", "_"))

    torch.save({
        'model': model,
        'dataset_config': dataset_config,
        'num_channels': num_channels,
        'num_classes': num_classes,
        'model_state_dict': model_state_dict,
        **_add_additional_parameters(model, wavelengths, spatial_size)
    },
        path
    )

def load_model(path: str):
    if not os.path.exists(path):
        return None, None

    checkpoint = torch.load(path)
    model =  get_model(
        model=checkpoint['model'],
        num_channels=checkpoint['num_channels'],
        num_classes=checkpoint['num_classes'],
        wavelengths=checkpoint['wavelengths'],
        spatial_size=checkpoint['spatial_size']
        
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['dataset_config']


def get_model_size(model: str):
    if  model == 'svm' or model == 'pls_da':
        return 1

    channels = 200
    config = DEFAULT_HPARAMS[model]

    if 'components' in config.keys():
        channels = config['components']

    model = get_model(model, channels, 20, np.arange(0, 200))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return int(params)


if __name__ == '__main__':
    for m in VALID_MODELS:
        print(f"{m}\t{get_model_size(m)}")
