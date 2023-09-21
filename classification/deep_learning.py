import argparse
import copy
import warnings
from torch import nn
from torch.utils.data import DataLoader
import time
from dataloader.dataset_factory import get_data, DataObject
from dataloader.valid_dataset_configs import *
from evaluation import evaluate_predictions_on_test_set
from evaluation.api import report_model_performance, init_evaluation_database
from sample_transforms.augment import *
from model_factory import get_model, VALID_MODELS, get_default_model_hparams, save_model, load_model, \
    get_pretrained_model, VALID_MODELS_PRETRAINING
from sample_transforms.pca import PCA
import os


def get_args():
    parser = argparse.ArgumentParser("DeepHS classification:")

    parser.add_argument('data_set_root')

    parser.add_argument('--model', type=str, choices=VALID_MODELS.keys(), default='deephs_net')

    parser.add_argument('--pca', type=lambda s: True if s.lower() == 'true' else False, default=None)
    parser.add_argument('--components', type=int, default=None)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)

    parser.add_argument('--store_checkpoints', type=str, default=None)
    parser.add_argument('--load_pretrained', type=str, default=None)

    args = parser.parse_args()

    return args


def reset_model_hparams(hparams):
    model = hparams['model']
    default_hparams = get_default_model_hparams(model)

    for p in default_hparams.keys():
        if hparams[p] is not None and hparams[p] != default_hparams[p]:
            warnings.warn(
                'Model specific default {}={} overwritten by {}'.format(p, default_hparams[p], hparams[p]))
        else:
            hparams[p] = default_hparams[p]

    if hparams['pca']:
        hparams['components'] = 30 if hparams['components'] is None else hparams['components']
    else:
        hparams['components'] = None

    return hparams


def _default_collate_fn(batch, labels=True):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] if torch.is_tensor(b[1]) else torch.tensor(b[1]) for b in batch]) if labels \
        else [None for b in batch]
    metas = [b[2] for b in batch]

    return xs, ys, metas


def get_data_loaders(data: DataObject, hparams, collate_fn=_default_collate_fn):
    train_loader = DataLoader(data.datasets.train, collate_fn=collate_fn, batch_size=hparams['batch_size'],
                              num_workers=hparams['num_workers'], shuffle=True, drop_last=True)
    val_loader = DataLoader(data.datasets.val, collate_fn=collate_fn, batch_size=1,
                            num_workers=hparams['num_workers'], shuffle=False, drop_last=True)
    test_loader = DataLoader(data.datasets.test, collate_fn=lambda batch: collate_fn(batch, labels=False),
                             batch_size=hparams['batch_size'], num_workers=hparams['num_workers'], drop_last=False)

    return train_loader, val_loader, test_loader


def get_loss_fn(loss: str):
    if loss == 'CE':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

def get_optimizer(optimizer: str, scheduler: str, model: nn.Module, hparams, split_lr=False):
    if optimizer == 'SGD':
        if split_lr:
            optimizer = torch.optim.SGD([{'params': model.get_backbone().parameters(), 'lr': hparams['lr']},
                                         {'params': model.get_head().parameters(), 'lr': hparams['lr'] * 10}],
                                        lr=hparams['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'])
    elif optimizer == 'Adam':
        if split_lr:
            optimizer = torch.optim.Adam([{'params': model.get_backbone().parameters(), 'lr': hparams['lr']},
                                         {'params': model.get_head().parameters(), 'lr': hparams['lr'] * 10}],
                                         lr=hparams['lr'], weight_decay=1e-06)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=1e-06)
    else:
        raise NotImplementedError

    if scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    else:
        raise NotImplementedError

    return optimizer, scheduler


def train(model: nn.Module,
          train_loader: DataLoader, val_loader: DataLoader,
          optimizer, scheduler, criterion,
          hparams,
          pca: PCA = None):
    val_loss_min = np.inf
    val_loss_last = np.inf
    epochs = 0
    patience = 10

    for t in range(hparams['epochs']):

        # training
        model.train()

        train_loss = 0.0
        for x, y, meta in train_loader:
            if pca is not None:
                x = pca(x)
            x, y = x.to(hparams['device']), y.to(hparams['device'])
            optimizer.zero_grad()
            y_pred = model(x, meta_data=meta)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()

        val_loss = 0.0
        for x, y, meta in val_loader:
            if pca is not None:
                x = pca(x)
            x, y = x.to(hparams['device']), y.to(hparams['device'])
            y_pred = model(x, meta_data=meta)
            loss = criterion(y_pred, y)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        print('Epoch {} \t\t Train loss: {:.4f}, val loss: {:.6f}'.format(t + 1, train_loss, val_loss))

        # checkpoint (min. val loss)
        if val_loss_min > val_loss:
            print('Val loss decreased --> saving model')
            val_loss_min = val_loss
            best_model = copy.deepcopy(model)

        # early stopping (val loss decrease)
        if val_loss_last > val_loss:
            epochs = 0
        else:
            epochs += 1
            if epochs == patience:
                print('Val loss not decreased for last {} epochs --> stopping'.format(patience))
                break
        val_loss_last = val_loss

        scheduler.step()

    return best_model


@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader, hparams, pca: PCA = None):
    model.eval()

    prediction = []
    for x, _, meta in test_loader:
        if pca is not None:
            x = pca(x)
        y_pred = model(x.to(hparams['device']), meta_data=meta).argmax(-1).cpu().detach()
        prediction += [y_.item() for y_ in y_pred]

    return np.asarray(prediction)


if __name__ == '__main__':
    start_ts = time.time()
    print('### Model Training + Eval. ###')

    # hyperparameters
    args = get_args()
    hparams = vars(args)
    hparams = reset_model_hparams(hparams)
    hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['num_workers'] = 8
    print('Hyperparameters: {}'.format(hparams))

    if hparams['store_checkpoints'] is not None:
        os.makedirs(hparams['store_checkpoints'], exist_ok=True)

    init_evaluation_database(hparams=hparams)
    for config in VALID_DATASET_CONFIG:
        print('\nDatset config. {}'.format(config))

        # get data
        train_augmentations = [RandomFlip(0.5), RandomRotate(0.5), RandomCut(0.5), RandomCrop(0.1)]
        data = get_data(config, augmentations=train_augmentations, data_set_root=hparams['data_set_root'])

        if data is None:
            continue

        train_loader, val_loader, test_loader = get_data_loaders(data, hparams)

        # PCA
        # pca = PCA(data_raw, n_components=hparams['components']) if hparams['pca'] else None
        # num_channels = hparams['components'] if hparams['pca'] else len(data.info.channels)
        if hparams['pca']:
            data_raw = get_data(config, data_set_root=hparams['data_set_root'])
            pca = PCA(data_raw, n_components=hparams['components'])
            num_channels = hparams['components']
        else:
            pca = None
            num_channels = len(data.info.channels)

        seeds = [0, 2, 5]
        for seed in seeds:
            print('Seed {}:'.format(seed))
            torch.manual_seed(seed)

            # setup
            if hparams['load_pretrained'] is None:
                model = get_model(hparams['model'],
                                  num_channels=num_channels,
                                  num_classes=len(data.info.classes),
                                  wavelengths=data.info.channels,
                                  spatial_size=data.info.image_size[0],
                                  )
            else:
                assert hparams['model'] in VALID_MODELS_PRETRAINING
                # model = load_model(hparams['load_pretrained']) # 'regular' model loading only working for config = pretraining_config
                model = get_pretrained_model(hparams['model'],
                                                        path=hparams['load_pretrained'],
                                                        num_channels=num_channels,
                                                        num_classes=len(data.info.classes),  # reset head
                                                        wavelengths=None, # do not reset first Hyve Conv layer nor Gaussians
                                                        spatial_size=data.info.image_size[0],
                                                        reset_seed=seed)#, reset_BN=True)
                #model.reset_first_layer(wavelength_range=wavelength_range, seed=seed)
            model = model.to(hparams['device'])

            criterion = get_loss_fn(hparams['loss'])
            optimizer, scheduler = get_optimizer(hparams['optimizer'], hparams['scheduler'], model, hparams)

            # train
            best_model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, hparams, pca)

            if hparams['store_checkpoints'] is not None:
                save_model(
                    model=hparams['model'],
                    dataset_config=config,
                    num_channels=num_channels,
                    num_classes=len(data.info.classes),
                    wavelengths=data.info.channels,
                    spatial_size=data.info.image_size[0],
                    model_state_dict=best_model.state_dict(),
                    output_path=hparams['store_checkpoints'],
                    name_addition=str(seed)
                )
                print("Best model checkpoint stored")

            # test
            prediction = test(best_model, test_loader, hparams, pca)
            print('Prediction: {}'.format(prediction))

            evaluate_predictions_on_test_set(config, prediction, random_seed=seed, data_set_root=hparams['data_set_root'])

    report_model_performance()

    print(f"## Took {time.time() - start_ts} s")
