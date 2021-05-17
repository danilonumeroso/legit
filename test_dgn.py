import typer
import torch
from utils import get_split, get_dgn
from torch_geometric.utils import precision, recall, f1_score, accuracy


def main(dataset_name: str, experiment_name: str):

    ts = get_split(dataset_name, 'test', experiment_name)
    dgn = get_dgn(dataset_name, experiment_name)

    Ypred = torch.stack([dgn(T.x, T.edge_index)[0].argmax() for T in ts])
    Ytrue = torch.stack([T.y for T in ts]).flatten()

    print(f"A: {accuracy(Ypred, Ytrue)}")
    print(f"P: {precision(Ypred, Ytrue, dgn.num_output).mean().item()}")
    print(f"R: {recall(Ypred, Ytrue, dgn.num_output).mean().item()}")
    print(f"F1: {f1_score(Ypred, Ytrue, dgn.num_output).mean().item()}")


if __name__ == "__main__":
    typer.run(main)
