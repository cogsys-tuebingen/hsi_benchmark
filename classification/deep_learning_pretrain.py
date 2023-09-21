from typing import Callable

import pandas as pd
import torchvision

from classification.model_factory import VALID_MODELS_PRETRAINING
from deep_learning import *
from classification.model_factory import save_model
from evaluation.api import init_evaluation_database
from torch.utils.data import ConcatDataset, RandomSampler
from dataloader.basic_dataloader import Dataset, HSDataset
import re

def get_args():
    parser = argparse.ArgumentParser("DeepHS pretraining + classification:")

    parser.add_argument('data_set_root')

    parser.add_argument('--pretrain_config', type=str, default='remote_sensing/salinas/0.3')
    parser.add_argument('--config', type=str, choices=VALID_DATASET_CONFIG, default='remote_sensing/indian_pines/0.05')

    parser.add_argument('--model', type=str, choices=VALID_MODELS_PRETRAINING, default='deephs_hyve_net')

    parser.add_argument('--pca', type=lambda s: True if s.lower() == 'true' else False, default=None)
    parser.add_argument('--components', type=int, default=None)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)

    parser.add_argument('--store_checkpoints', type=str, required=True) #store pretrained and final model
    parser.add_argument('--load_pretrained', type=str, default=None)

    args = parser.parse_args()

    return args


def _collate_fn(batch, labels=True):
    xs = [b[0] for b in batch]
    xs_shapes = torch.tensor([x.shape for x in xs])
    max_shape = xs_shapes.max(0)[0]
    min_shape = xs_shapes.min(0)[0]
    if (max_shape != min_shape).any():
        padded_xs = torch.ones((len(batch), *max_shape), dtype=xs[0].dtype, device=xs[0].device) * -255
        for i, x in enumerate(xs):
            padded_xs[i, :x.shape[0], :x.shape[1], :x.shape[2]] = x
        xs = padded_xs
    else:
        xs = torch.stack(xs)

    ys = torch.stack([b[1] if torch.is_tensor(b[1]) else torch.tensor(b[1]) for b in batch]) if labels \
        else [None for b in batch]
    metas = [b[2] for b in batch]

    return xs, ys, metas


# class DatasetWithClassOffset(Dataset):
#     '''
#     General structure: item (X), label (y), meta data
#     '''
#     def __init__(self, org_dataset: HSDataset, class_offset):
#         self.org_dataset = org_dataset
#         self.class_offset = class_offset
#
#     def __len__(self):
#         return len(self.org_dataset)
#
#     def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, object]:
#         x, y, meta_data = self.org_dataset[index]
#         return x, y+self.class_offset, meta_data


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    from https://towardsdatascience.com/unbalanced-data-loading-for-multi-task-learning-in-pytorch-e030ad5033b
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, mixed_batches=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.records) for cur_dataset in dataset.datasets])

        self.mixed_batches = mixed_batches

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        if not self.mixed_batches: # either = batch_size -> one config per batch
            samples_to_grab = self.batch_size
        else: # or mixed batches
            samples_to_grab = self.batch_size // self.number_of_datasets  # -> combine the tasks in a balanced way, and by setting the samples_to_grab to 4, e.g., which is half of the batch size, we can get a mixed mini-batch with 4 samples taken from each task. To produce a ratio of 1:2 toward a more important task, we can set samples_to_grab=2 for the first task and samples_to_grab=6 for the second task
        epoch_samples = self.largest_dataset_size * self.number_of_datasets
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


def get_multi_data_loaders(data: DataObject, hparams, collate_fn, mixed_batches=False):
    train_sampler = BatchSchedulerSampler(dataset=data.datasets.train, batch_size=hparams['batch_size'], mixed_batches=mixed_batches)
    val_sampler = BatchSchedulerSampler(dataset=data.datasets.val, batch_size=hparams['batch_size'])

    train_loader = DataLoader(data.datasets.train, sampler=train_sampler, collate_fn=collate_fn, batch_size=hparams['batch_size'],
                              num_workers=hparams['num_workers'], shuffle=False)
    # val_loader = DataLoader(data.datasets.val, collate_fn=collate_fn, batch_size=hparams['batch_size'],
    #                         num_workers=hparams['num_workers'], shuffle=False)
    val_loader = DataLoader(data.datasets.val, sampler=val_sampler, collate_fn=collate_fn, batch_size=hparams['batch_size'],
                            num_workers=hparams['num_workers'], shuffle=False)
    test_loader = DataLoader(data.datasets.test, collate_fn=lambda batch: collate_fn(batch, labels=False),
                             batch_size=hparams['batch_size'], num_workers=hparams['num_workers'], drop_last=False)

    return train_loader, val_loader, test_loader


def get_pretrain_data(pretrain_config, train_augmentations, hparams, mixed_batches=False):
    if isinstance(pretrain_config, list):
        print('Load multiple pretraining data sets')

        pretrain_datas = [get_data(p_config, augmentations=train_augmentations, data_set_root=hparams['data_set_root']) for p_config in pretrain_config]

        num_classes = max([len(pd.info.classes) for pd in pretrain_datas])
        extreme_channels = [min([min(pd.info.channels) for pd in pretrain_datas]), max([max(pd.info.channels) for pd in pretrain_datas])]
        channel_wavelengths = extreme_channels
        num_channels = max([len(pd.info.channels) for pd in pretrain_datas])
        spatial_size = max([pd.info.image_size for pd in pretrain_datas])

        combined_train_dataset = []
        combined_val_dataset = []
        combined_test_dataset = []

        # class_offset = 0
        # for pd in pretrain_datas:
        #     if class_offset == 0:
        #         combined_train_dataset.append(pd.datasets.train)
        #         combined_val_dataset.append(pd.datasets.val)
        #         combined_test_dataset.append(pd.datasets.test)
        #     else:
        #         combined_train_dataset.append(DatasetWithClassOffset(pd.datasets.train, class_offset))
        #         combined_val_dataset.append(DatasetWithClassOffset(pd.datasets.val, class_offset))
        #         combined_test_dataset.append(DatasetWithClassOffset(pd.datasets.test, class_offset))
        #     class_offset = len(pd.info.classes)

        # num_classes = 0
        # previous_classes = []
        # # channel_wavelengths = set([]) # alternative 1
        # for pd in pretrain_datas:
        #     # channel_wavelengths = channel_wavelengths | set(pd.info.channels) # alternative 1
        #     if pd.info.classes in previous_classes:
        #         combined_train_dataset.append(pd.datasets.train)
        #         combined_val_dataset.append(pd.datasets.val)
        #         combined_test_dataset.append(pd.datasets.test)
        #     else:
        #         combined_train_dataset.append(DatasetWithClassOffset(pd.datasets.train, class_offset=num_classes))
        #         combined_val_dataset.append(DatasetWithClassOffset(pd.datasets.val, class_offset=num_classes))
        #         combined_test_dataset.append(DatasetWithClassOffset(pd.datasets.test, class_offset=num_classes))
        #         previous_classes.append(pd.info.classes)
        #         num_classes += len(pd.info.classes)

        for pd in pretrain_datas:
            combined_train_dataset.append(pd.datasets.train)
            combined_val_dataset.append(pd.datasets.val)
            combined_test_dataset.append(pd.datasets.test)

        combined_train_dataset = ConcatDataset(combined_train_dataset)
        combined_val_dataset = ConcatDataset(combined_val_dataset)
        combined_test_dataset = ConcatDataset(combined_test_dataset)

        # channel_wavelengths = [300, 3000]
        # extreme_channels = [300, 3000]
        # num_channels = 0

        # extreme_channels = [min(channel_wavelengths), max(channel_wavelengths)]
        # num_channels = len(channel_wavelengths)

        pretrain_data = DataObject("combined", combined_train_dataset,
                                   combined_val_dataset, combined_test_dataset,
                                   pretrain_datas[0].info.task_type,
                                   classes=None, channels=None, image_size=spatial_size)
        train_loader, val_loader, test_loader = get_multi_data_loaders(pretrain_data, hparams, collate_fn=_collate_fn,
                                                                       mixed_batches=mixed_batches)
    else:
        print('Load single pretraining data set')
        pretrain_data = get_data(pretrain_config, augmentations=train_augmentations, data_set_root=hparams['data_set_root'])
        train_loader, val_loader, test_loader = get_data_loaders(pretrain_data, hparams)
        num_channels = len(pretrain_data.info.channels)
        num_classes = len(pretrain_data.info.classes)
        channel_wavelengths = pretrain_data.info.channels
        spatial_size = pretrain_data.info.image_size[0]

    return pretrain_data, (train_loader, val_loader, test_loader), num_classes, num_channels, channel_wavelengths, spatial_size


def run_without_pretraining(config, hparams, seed=0, epochs=50,
                            multitask=False):
    hparams['epochs'] = epochs
    print('Hyperparameters: {}'.format(hparams))

    ## training:
    # get data
    train_augmentations = [RandomFlip(0.5), RandomRotate(0.5), RandomCut(0.5), RandomCrop(0.1)]
    data = get_data(config, augmentations=train_augmentations, data_set_root=hparams['data_set_root'])
    train_loader, val_loader, test_loader = get_data_loaders(data, hparams)

    # setup
    pca = PCA(data, n_components=hparams['components']) if hparams['pca'] else None
    num_channels = hparams['components'] if hparams['pca'] else len(data.info.channels)
    num_classes = len(data.info.classes)
    channel_wavelengths = data.info.channels
    spatial_size = data.info.image_size[0]

    model = get_model(hparams['model'],
                      num_channels=num_channels,
                      num_classes=num_classes,
                      wavelengths=channel_wavelengths,
                      spatial_size=spatial_size,
                      multitask=multitask)
    model = model.to(hparams['device'])

    model.reset(seed=seed)

    criterion = get_loss_fn(hparams['loss'])
    optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], model, hparams)

    # train
    final_model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)

    # save model
    path = os.path.join(hparams['store_checkpoints'], hparams['model'] + '.pt')
    torch.save({
        'model': hparams['model'],
        'dataset_config': config,
        'num_channels': num_channels,
        'num_classes': num_classes,
        'wavelengths': channel_wavelengths,
        'model_state_dict': final_model.state_dict(),
        'spatial_size': spatial_size
    },
        path
    )
    print('Save model to {}'.format(path))

    ## evaluation:
    prediction = test(final_model, test_loader, hparams, pca)
    print('Prediction: {}'.format(prediction))

    evaluate_predictions_on_test_set(config, prediction, data_set_root=hparams['data_set_root'], random_seed=seed)


def run_with_pretraining(pretrain_config, config, hparams, seed=0,
                         epochs_pretraining=50, epochs_finetuning=50,
                         freeze_backbone=False, reduce_lr=None, diff_lrs=False,
                         reset_wavelengths=False, reset_BN=False,
                         mixed_batches=True,
                         multitask=True):
    train_augmentations = [RandomFlip(0.5), RandomRotate(0.5), RandomCut(0.5), RandomCrop(0.1)]
    old_lr = hparams['lr']

    # pretraining:
    multitask_pretraining = False
    if hparams['load_pretrained'] is None:
        hparams['epochs'] = epochs_pretraining

        # get data
        pretrain_data, loaders, num_classes, num_channels, channel_wavelengths, spatial_size = get_pretrain_data(pretrain_config, train_augmentations, hparams, mixed_batches)
        train_loader, val_loader, _ = loaders

        # setup
        pca = PCA(pretrain_data, n_components=hparams['components']) if hparams['pca'] else None
        num_channels = hparams['components'] if hparams['pca'] else num_channels

        multitask_pretraining = isinstance(pretrain_config, list)  # get multitask model for pretraining on multiple configs
        model = get_model(hparams['model'],
                          num_channels=num_channels,
                          num_classes=num_classes,
                          wavelengths=channel_wavelengths,
                          spatial_size=spatial_size,
                          multitask=multitask_pretraining)
        model = model.to(hparams['device'])

        model.reset(seed=seed)

        criterion = get_loss_fn(hparams['loss'])
        optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], model, hparams)

        hparams['lr'] = hparams['lr'] / len(pretrain_config) if multitask_pretraining else old_lr # reduce lr / # configs for multitask model # TODO

        # train
        pretrained_model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)

        # save pretrained model
        path = os.path.join(hparams['store_checkpoints'], hparams['model'] + '_pretrained.pt')
        torch.save({
            'model': hparams['model'],
            'dataset_config': pretrain_config,
            'num_channels': num_channels,
            'num_classes': num_classes,
            'wavelengths': channel_wavelengths,
            'model_state_dict': pretrained_model.state_dict(),
            'spatial_size': spatial_size
        },
            path
        )
        print('Save pretrained model to {}'.format(path))

    ## actual training / fine-tuning:
    # get data for training (fine-tuning)
    data = get_data(config, augmentations=train_augmentations, data_set_root=hparams['data_set_root'])
    train_loader, val_loader, test_loader = get_data_loaders(data, hparams)

    # setup
    pca = PCA(data, n_components=hparams['components']) if hparams['pca'] else None
    num_channels = hparams['components'] if hparams['pca'] else len(data.info.channels)
    channel_wavelengths = data.info.channels
    if reset_wavelengths:
        wavelength_range = (min(channel_wavelengths), max(channel_wavelengths))
    num_classes = len(data.info.classes)

    spatial_size = data.info.image_size[0]

    if multitask_pretraining and not multitask:
        # after multi-task pretraining: load multitask model backbone weights into regular model
        hparams['load_pretrained'] = os.path.join(hparams['store_checkpoints'], hparams['model'] + '_pretrained.pt')

    if hparams['load_pretrained'] is not None:
        # load pretrained model
        pretrained_model = get_pretrained_model(hparams['model'],
                                                path=hparams['load_pretrained'],
                                                num_channels=num_channels,
                                                num_classes=num_classes,  # reset head
                                                wavelengths=None,
                                                # do not reset first Hyve Conv layer nor Gaussians
                                                spatial_size=spatial_size,
                                                reset_seed=seed, reset_BN=reset_BN)
    else:
        pretrained_model.reset_head(num_classes=num_classes, seed=seed, BN=reset_BN)

    if reset_wavelengths:
        pretrained_model.reset_first_layer(wavelength_range=wavelength_range, seed=seed)
    pretrained_model = pretrained_model.to(hparams['device'])

    # train
    hparams['lr'] = old_lr
    criterion = get_loss_fn(hparams['loss'])
    if freeze_backbone:
        # a) freeze backbone and train FC head only (few epochs)
        print('Step 1: Freeze backbone and train adapted first layer + head only')
        hparams['epochs'] = 20
        hparams['lr'] = hparams['lr'] * 10
        pretrained_model.freeze_backbone()
        optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], pretrained_model, hparams)
        intermediate_model = train(pretrained_model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)

        # b) fine-tune whole model (with lower lr)
        print('Step 2: Fine-tune whole model')
        intermediate_model.unfreeze_backbone()
        hparams['lr'] = hparams['lr'] / 10. # original lr
        hparams['epochs'] = epochs_finetuning
        optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], intermediate_model, hparams)
        final_model = train(intermediate_model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)
    else:
        hparams['epochs'] = epochs_finetuning
        if reduce_lr is not None:
            hparams['lr'] = hparams['lr'] / float(reduce_lr)
        optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], pretrained_model, hparams, split_lr=diff_lrs)
        final_model = train(pretrained_model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)

    # save final model
    path = os.path.join(hparams['store_checkpoints'], hparams['model'] + '_finetuned.pt')
    torch.save({
        'model': hparams['model'],
        'dataset_config': config,
        'num_channels': num_channels,
        'num_classes': num_classes,
        'wavelengths': channel_wavelengths,
        'model_state_dict': final_model.state_dict(),
        'spatial_size': spatial_size
    },
        path
    )
    print('Save final model to {}'.format(path))

    ## evaluation:
    prediction = test(final_model, test_loader, hparams, pca)
    print('Prediction: {}'.format(prediction))

    evaluate_predictions_on_test_set(config, prediction, data_set_root=hparams['data_set_root'], random_seed=seed)


if __name__ == '__main__':
    args = get_args()
    hparams = vars(args)
    hparams = reset_model_hparams(hparams)
    hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['num_workers'] = 8
    seeds = [0, 5, 1337]

    assert hparams['model'] in VALID_MODELS_PRETRAINING

    pretrain_config = hparams['pretrain_config'].split(";") if ';' in hparams['pretrain_config'] else hparams['pretrain_config']
    config = hparams['config']
    print('\nDatset configs. {} (pretraining), {} (fine-tuning + testing)'.format(pretrain_config, config))

    reports = []
    modes = []

    init_evaluation_database(hparams=hparams)

    print('## Without pretraining: ')
    modes.append('Without pretraining')
    for seed in seeds:
        run_without_pretraining(config, hparams, seed=seed)
    reports.append(report_model_performance(short=True))

    init_evaluation_database(hparams=hparams)

    print('## Without pretraining (muti-task model): ')
    modes.append('Without pretraining (multi-task model)')
    for seed in seeds:
        run_without_pretraining(config, hparams, seed=seed, multitask=True)
    reports.append(report_model_performance(short=True))

    init_evaluation_database(hparams=hparams)

    print('## With pretraining: ')
    modes.append('With pretraining')
    for seed in seeds:
        run_with_pretraining(pretrain_config, config, hparams, seed=seed)
    reports.append(report_model_performance(short=True))

    init_evaluation_database(hparams=hparams)

    print('## With pretraining (no multi-task): ')
    modes.append('With pretraining (no multi-task)')
    for seed in seeds:
        run_with_pretraining(pretrain_config, config, hparams, seed=seed, multitask=False)
    reports.append(report_model_performance(short=True))

    # init_evaluation_database(hparams=hparams)
    #
    # print('## With pretraining (reinit BN): ')
    # modes.append('With pretraining (reinit BN)')
    # for seed in seeds:
    #     run_with_pretraining(pretrain_config, config, hparams, seed=seed, reset_BN=True)
    # reports.append(report_model_performance(short=True))
    #
    # init_evaluation_database(hparams=hparams)
    #
    # print('## With pretraining (reinit Gaussians): ')
    # modes.append('With pretraining (reinit Gaussians)')
    # for seed in seeds:
    #     run_with_pretraining(pretrain_config, config, hparams, seed=seed, reset_wavelengths=True)
    # reports.append(report_model_performance(short=True))

    # init_evaluation_database(hparams=hparams)
    #
    # print('## With pretraining (diff. lrs): ')
    # modes.append('With pretraining (diff. lrs)')
    # for seed in seeds:
    #     run_with_pretraining(pretrain_config, config, hparams, seed=seed, diff_lrs=True)
    # reports.append(report_model_performance(short=True))
    #
    # init_evaluation_database(hparams=hparams)
    #
    # print('## With pretraining (freeze backbone): ')
    # modes.append('With pretraining (freeze backbone)')
    # for seed in seeds:
    #     run_with_pretraining(pretrain_config, config, hparams, seed=seed, freeze_backbone=True)
    # reports.append(report_model_performance(short=True))

    print("### Final report ###")
    print('\nDatset configs. {} (pretraining), {} (fine-tuning + testing)'.format(pretrain_config, config))
    for variant, report in zip(modes, reports):
        print(variant)
        print(report)
