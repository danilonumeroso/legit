import torch
import random

from torch_geometric.data import DataLoader
from torch.nn import functional as F
from utils.molecules import check_molecule_validity, pyg_to_mol_tox21
from torch_geometric.datasets import TUDataset, MoleculeNet
from utils.cycliq import CYCLIQ


def pre_transform(sample, n_pad):
    sample.x = F.pad(sample.x, (0, n_pad), "constant")
    return sample


def get_split(dataset_name, split, experiment):

    if dataset_name.lower() == 'tox21':
        ds = TUDataset('data/tox21',
                       name='Tox21_AhR_testing',
                       pre_transform=lambda sample: pre_transform(sample, 2))

    elif dataset_name.lower() == 'esol':

        ds = MoleculeNet('data/esol', name='ESOL')

    elif dataset_name.lower() == 'cycliq':
        ds = CYCLIQ('data/cycliq', name='CYCLIQ-TS')

    ds.data, ds.slices = torch.load(
        f"runs/{dataset_name.lower()}/{experiment}/splits/{split}.pth")

    return ds


def preprocess(dataset_name, experiment_name, batch_size):
    return _PREPROCESS[dataset_name.lower()](experiment_name, batch_size)


def _preprocess_tox21(experiment_name, batch_size):

    dataset_tr = TUDataset(
        'data/tox21',
        name='Tox21_AhR_training',
        pre_transform=lambda sample: pre_transform(sample, 3))

    dataset_vl = TUDataset(
        'data/tox21',
        name='Tox21_AhR_evaluation',
        pre_transform=lambda sample: pre_transform(sample, 0))

    dataset_ts = TUDataset(
        'data/tox21',
        name='Tox21_AhR_testing',
        pre_transform=lambda sample: pre_transform(sample, 2))

    data_list = ([dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
                 [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
                 [dataset_ts.get(idx) for idx in range(len(dataset_ts))])

    data_list = list(
        filter(lambda mol: check_molecule_validity(mol, pyg_to_mol_tox21),
               data_list))

    POSITIVES = list(filter(lambda x: x.y == 1, data_list))
    NEGATIVES = list(filter(lambda x: x.y == 0, data_list))
    N_POSITIVES = len(POSITIVES)
    N_NEGATIVES = N_POSITIVES
    NEGATIVES = NEGATIVES[:N_NEGATIVES]

    data_list = POSITIVES + NEGATIVES
    random.shuffle(data_list)

    n = len(data_list) // 10
    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]

    train = dataset_tr
    val = dataset_vl
    test = dataset_ts

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices),
               f'runs/tox21/{experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices),
               f'runs/tox21/{experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices),
               f'runs/tox21/{experiment_name}/splits/test.pth')

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


def _preprocess_esol(experiment_name, batch_size):

    dataset = MoleculeNet('data/esol', name='ESOL')

    data_list = ([dataset.get(idx) for idx in range(len(dataset))])

    random.shuffle(data_list)

    n = len(data_list) // 10

    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]
    train_data = train_data[n:]

    train = dataset
    val = dataset.copy()
    test = dataset.copy()

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices),
               f'runs/esol/{experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices),
               f'runs/esol/{experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices),
               f'runs/esol/{experiment_name}/splits/test.pth')

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


def _preprocess_cycliq(experiment_name, batch_size):
    return _cycliq("CYCLIQ", experiment_name, batch_size)


def _cycliq(name, experiment_name, batch_size):

    dataset_tr = CYCLIQ('data/cycliq',
                        graphs_distribution={
                            'dims': (35, 55),
                            'count': (1, 2),
                            'cycle_len': (4, 6),
                            'clique_size': (3, 7)
                        },
                        name='CYCLIQ-TR')

    dataset_ts = CYCLIQ(root='data/cycliq',
                        sample_size=100,
                        graphs_distribution={
                            'dims': (35, 55),
                            'count': (1, 2),
                            'cycle_len': (8, 12),
                            'clique_size': (3, 5)
                        },
                        name='CYCLIQ-TS')

    data_list = [dataset_tr.get(idx) for idx in range(len(dataset_tr))]

    for i, d in enumerate(data_list):
        d.gexf_id = f"{i+1}.{d.y.item()}.gexf"

    random.shuffle(data_list)

    n = len(data_list) // 10

    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = [dataset_ts.get(idx) for idx in range(len(dataset_ts))]

    train = dataset_tr
    val = dataset_tr.copy()
    test = dataset_ts
    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices),
               f'runs/{name.lower()}/{experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices),
               f'runs/{name.lower()}/{experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices),
               f'runs/{name.lower()}/{experiment_name}/splits/test.pth')

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


_PREPROCESS = {
    'tox21': _preprocess_tox21,
    'esol': _preprocess_esol,
    'cycliq': _preprocess_cycliq
}
