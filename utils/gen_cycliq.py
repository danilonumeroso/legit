import random
from enum import Enum
from pathlib import Path

import networkx as nx
import typer
from tqdm import tqdm as tq


class Dataset(str, Enum):
    CYCLIQ = 'CYCLIQ'
    CYCLIQ_MULTI = 'CYCLIQ-MULTI'


def random_tree(n):
    g = nx.generators.trees.random_tree(n)
    for i in range(n):
        g.nodes[i]['label'] = 0
    return g


def attach_cycle(g, cycle_len, label, is_clique):
    N = len(g.nodes())
    host_cands = [k for k, v in g.nodes(data=True) if v['label'] == 0]
    host_node = random.choice(host_cands)
    neighbors = list(g.neighbors(host_node))
    for u in neighbors:
        g.remove_edge(u, host_node)

    # add the cycle
    cycle_nodes = [host_node]
    for i in range(cycle_len - 1):
        g.add_edge(cycle_nodes[-1], N + i)
        cycle_nodes.append(N + i)
    g.add_edge(host_node, cycle_nodes[-1])

    if is_clique:
        for u in cycle_nodes:
            for v in cycle_nodes:
                if u != v:
                    g.add_edge(u, v)

    for u in cycle_nodes:
        g.nodes[u]['label'] = label

    # restore host_node edges
    for u in neighbors:
        v = random.choice(cycle_nodes)
        g.add_edge(u, v)
    return g


def attach_cycles(g, cycle_len, count, is_clique=False):
    for i in range(count):
        attach_cycle(g, cycle_len, '%d-%d-%d' % (cycle_len, is_clique, i), is_clique)
    return g


def add_to_list(graph_list, g, label):
    graph_num = len(graph_list) + 1
    for u in g.nodes():
        g.nodes()[u]['graph_num'] = graph_num
    g.graph['graph_num'] = graph_num
    graph_list.append((g, label))

def cycliq(sample_size, is_multi):
    all_graphs = []
    label = 0
    if is_multi:
        random.seed(0)
        for i in range(sample_size):
            g = random_tree(random.randint(8, 15))
            add_to_list(all_graphs, g, label)
        label += 1
    random.seed(1)
    for i in range(sample_size):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 2)
        attach_cycles(g, cycle_len=5, count=count)
        add_to_list(all_graphs, g, label)
    label += 1
    random.seed(2)
    for i in range(sample_size):
        g = random_tree(random.randint(8, 15))
        count = random.randint(1, 2)
        attach_cycles(g, cycle_len=5, count=count, is_clique=True)
        add_to_list(all_graphs, g, label)
    label += 1
    if is_multi:
        random.seed(3)
        for i in range(sample_size):
            g = random_tree(random.randint(8, 15))
            count = random.randint(1, 2)
            attach_cycles(g, cycle_len=5, count=count, is_clique=True)
            count = random.randint(1, 2)
            attach_cycles(g, cycle_len=5, count=count)
            add_to_list(all_graphs, g, label)
    return all_graphs


def write_gexf(output_path: Path, graphs):
    print('Created .gexf files in %s' % output_path)
    for g, label in graphs:
        nx.write_gexf(g, output_path / ('%d.%d.gexf' % (g.graph['graph_num'], label)))


def write_adjacency(output_path: Path, name: str, graphs):
    relabled_gs = []
    first_label = 1
    graph_indicator = []
    for g, label in tq(graphs):
        relabled_gs.append(nx.convert_node_labels_to_integers(g, first_label=first_label))
        N = len(g.nodes())
        first_label += N
        graph_indicator.extend([g.graph['graph_num']] * N)
    with open(output_path / ('%s_A.txt' % name), 'w') as f:
        for g in relabled_gs:
            for u, v in g.edges():
                f.write(f'{u}, {v}\n{v}, {u}\n')
    with open(output_path / ('%s_graph_indicator.txt' % name), 'w') as f:
        f.write('\n'.join(map(str, graph_indicator)))
        f.write('\n')
    with open(output_path / ('%s_graph_labels.txt' % name), 'w') as f:
        f.write('\n'.join([str(label) for g, label in graphs]))
        f.write('\n')


def main(dataset: Dataset,
         name: str = typer.Argument(None),
         output_path: Path = typer.Argument('data', help='Output path for dataset'),
         sample_size: int = typer.Option(1000, help='Number of samples for each label to generate')):
    print('Generating %s dataset' % dataset.value)
    if dataset == Dataset.CYCLIQ:
        graphs = cycliq(sample_size, is_multi=False)
    elif dataset == Dataset.CYCLIQ_MULTI:
        graphs = cycliq(sample_size, is_multi=True)
    elif dataset == Dataset.TRISQ:
        graphs = trisq(sample_size)
    elif dataset == Dataset.HOUSE_CLIQ:
        graphs = house_cliq(sample_size)
    elif dataset == Dataset.GRID_CLIQ:
        graphs = grid_cliq(sample_size)
    elif dataset == Dataset.HOUSE_GRID:
        graphs = house_grid(sample_size)

    if not output_path.exists():
        output_path.mkdir()

    if name is None:
        name = dataset.value

    output_path = output_path / dataset.value.lower() / name
    output_path.mkdir(exist_ok=True, parents=True)

    gexf_path = output_path / 'gexf'
    gexf_path.mkdir(exist_ok=True)

    raw = output_path / 'raw'
    raw.mkdir(exist_ok=True)

    processed = output_path / 'processed'
    processed.mkdir(exist_ok=True)

    write_gexf(gexf_path, graphs)
    write_adjacency(raw, name, graphs)


if __name__ == '__main__':
    app = typer.Typer(add_completion=False)
    app.command()(main)
    app()
