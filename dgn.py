import torch
import os
import os.path as osp
import json
import typer
import numpy as np

from pathlib import Path
from models.encoder import GCNN
from utils import get_split, get_dgn, preprocess, train_cycle_classifier, train_cycle_regressor, set_seed, create_path
from torch_geometric.utils import precision, recall, f1_score, accuracy

app = typer.Typer(add_completion=False)


@app.command(name='embed')
def embed(dataset_name: str, experiment_name: str):

    path = Path('./runs') / dataset_name / experiment_name / 'embeddings'
    create_path(path)
    create_path(path / 'train')
    create_path(path / 'test')

    tr = get_split(dataset_name, 'train', experiment_name)
    ts = get_split(dataset_name, 'test', experiment_name)
    dgn = get_dgn(dataset_name, experiment_name)

    for i, data in enumerate(tr):
        _, (node_emb, _) = dgn(data.x, data.edge_index)
        node_emb = node_emb.numpy()
        np.save(open(path / 'train' / f'{i+1}.npy', 'wb'), node_emb)

    for i, data in enumerate(ts):
        _, (node_emb, _) = dgn(data.x, data.edge_index)
        node_emb = node_emb.numpy()
        np.save(open(path / 'test' / f'{i+1}.npy', 'wb'), node_emb)


@app.command(name='test')
def test(dataset_name: str, experiment_name: str):

    ts = get_split(dataset_name, 'test', experiment_name)
    dgn = get_dgn(dataset_name, experiment_name)

    Ypred = torch.stack([dgn(T.x, T.edge_index)[0].argmax() for T in ts])
    Ytrue = torch.stack([T.y for T in ts]).flatten()

    print(f"A: {accuracy(Ypred, Ytrue)}")
    print(f"P: {precision(Ypred, Ytrue, dgn.num_output).mean().item()}")
    print(f"R: {recall(Ypred, Ytrue, dgn.num_output).mean().item()}")
    print(f"F1: {f1_score(Ypred, Ytrue, dgn.num_output).mean().item()}")


@app.command(name='train')
def train(dataset_name: str,
          experiment_name: str = typer.Argument("test"),
          lr: float = typer.Option(0.01),
          hidden_size: int = typer.Option(32),
          batch_size: int = typer.Option(32),
          dropout: float = typer.Option(0.1),
          epochs: int = typer.Option(50),
          seed: int = typer.Option(0)):

    set_seed(seed)

    dataset_name = dataset_name.lower()

    base_path = './runs/' + dataset_name + '/' + experiment_name
    if not osp.exists(base_path):
        os.makedirs(base_path + "/ckpt")
        os.makedirs(base_path + "/plots")
        os.makedirs(base_path + "/splits")
        os.makedirs(base_path + "/gen_output")
    else:
        import shutil
        shutil.rmtree(base_path + "/plots", ignore_errors=True)
        os.makedirs(base_path + "/plots")

    train_loader, val_loader, test_loader, *extra = preprocess(
        dataset_name, experiment_name, batch_size)
    train_ds, val_ds, test_ds, num_features, num_classes = extra

    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCNN(num_input=num_features,
                 num_hidden=hidden_size,
                 num_output=num_classes,
                 dropout=dropout).to(device)

    with open(base_path + '/hyperparams.json', 'w') as outfile:
        json.dump(
            {
                'num_input': num_features,
                'num_hidden': hidden_size,
                'num_output': num_classes,
                'dropout': dropout,
                'batch_size': batch_size,
                'lr': lr,
                'seed': seed
            }, outfile)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if dataset_name.lower() in ['tox21', 'cycliq', 'cycliq-multi']:
        train_cycle_classifier(task=dataset_name.lower(),
                               train_loader=train_loader,
                               val_loader=val_loader,
                               test_loader=test_loader,
                               len_train=len_train,
                               len_val=len_val,
                               len_test=len_test,
                               model=model,
                               optimizer=optimizer,
                               device=device,
                               base_path=base_path,
                               epochs=epochs)

    elif dataset_name.lower() in ['esol']:
        train_cycle_regressor(task=dataset_name.lower(),
                              train_loader=train_loader,
                              val_loader=val_loader,
                              test_loader=test_loader,
                              len_train=len_train,
                              len_val=len_val,
                              len_test=len_test,
                              model=model,
                              optimizer=optimizer,
                              device=device,
                              base_path=base_path,
                              epochs=epochs)


if __name__ == '__main__':
    app()
