import random
import os.path as osp
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io.tu import split, read_file, coalesce
from torch_geometric.utils import remove_self_loops
from pathlib import Path


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
        attach_cycle(g, cycle_len, '%d-%d-%d' % (cycle_len, is_clique, i),
                     is_clique)
    return g


def add_to_list(graph_list, g, label):
    graph_num = len(graph_list) + 1
    for u in g.nodes():
        g.nodes()[u]['graph_num'] = graph_num
    g.graph['graph_num'] = graph_num
    graph_list.append((g, label))


def gen_cycliq(sample_size, distrib):
    all_graphs = []

    for i in range(sample_size):
        a, b = distrib['dims']
        g = random_tree(random.randint(a, b))

        a, b = distrib['count']
        count = random.randint(a, b)

        a, b = distrib['clique_size']
        k = random.randint(a, b)

        attach_cycles(g, cycle_len=k, count=count)
        add_to_list(all_graphs, g, 0)

    for i in range(sample_size):
        a, b = distrib['dims']
        g = random_tree(random.randint(a, b))

        a, b = distrib['count']
        count = random.randint(a, b)

        a, b = distrib['clique_size']
        k = random.randint(a, b)

        attach_cycles(g, cycle_len=k, count=count, is_clique=True)
        add_to_list(all_graphs, g, 1)

    return all_graphs


def write_gexf(output_path: Path, graphs):
    print('Created .gexf files in %s' % output_path)
    for g, label in graphs:
        nx.write_gexf(
            g, output_path / ('%d.%d.gexf' % (g.graph['graph_num'], label)))


def write_adjacency(output_path: Path, dataset: str, graphs):
    relabled_gs = []
    first_label = 1
    graph_indicator = []
    for g, label in graphs:
        relabled_gs.append(
            nx.convert_node_labels_to_integers(g, first_label=first_label))
        N = len(g.nodes())
        first_label += N
        graph_indicator.extend([g.graph['graph_num']] * N)
    with open(output_path / ('%s_A.txt' % dataset), 'w') as f:
        for g in relabled_gs:
            for u, v in g.edges():
                f.write(f'{u}, {v}\n{v}, {u}\n')
    with open(output_path / ('%s_graph_indicator.txt' % dataset), 'w') as f:
        f.write('\n'.join(map(str, graph_indicator)))
        f.write('\n')
    with open(output_path / ('%s_graph_labels.txt' % dataset), 'w') as f:
        f.write('\n'.join([str(label) for g, label in graphs]))
        f.write('\n')


def read_cycliq_data(folder, prefix):

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    x = torch.ones((edge_index.max().item() + 1, 10))

    edge_attr = torch.ones((edge_index.size(1), 5))

    y = read_file(folder, prefix, 'graph_labels', torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    return data, slices


class CYCLIQ(InMemoryDataset):
    def __init__(
            self,
            root,
            name,
            sample_size=1000,
            graphs_distribution={
                'dims': (8, 15),
                'count': (1, 2),
                'len_structure': (1, 4),
                'clique_size': (2, 4)
            },
            transform=None,
            pre_transform=None,
            pre_filter=None):
        self.name = name
        self.sample_size = sample_size
        self.graph_distrib = graphs_distribution
        self.root = root

        super(CYCLIQ, self).__init__(root, transform, pre_transform,
                                     pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def gexf_dir(self):
        name = 'gexf'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        gexf_dir = Path(self.gexf_dir)
        gexf_dir.mkdir()

        graphs = gen_cycliq(self.sample_size, self.graph_distrib)
        write_gexf(gexf_dir, graphs)
        write_adjacency(Path(self.raw_dir), self.name, graphs)

        pass

    def process(self):
        self.data, self.slices = read_cycliq_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
