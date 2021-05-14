import random
import networkx as nx

from pathlib import Path
from tqdm import tqdm as tq


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
        attach_cycle(g,
                     cycle_len,
                     '%d-%d-%d' % (cycle_len, is_clique, i),
                     is_clique)
    return g


def add_to_list(graph_list, g, label):
    graph_num = len(graph_list) + 1
    for u in g.nodes():
        g.nodes()[u]['graph_num'] = graph_num
    g.graph['graph_num'] = graph_num
    graph_list.append((g, label))


def cycliq(sample_size, graph_distrib):
    all_graphs = []
    label = 0

    distrib_0, distrib_1 = graph_distrib

    for i in range(sample_size):
        a, b = distrib_0['dims']
        g = random_tree(random.randint(a, b))

        a, b = distrib_0['count']
        count = random.randint(a, b)

        k = distrib_0['len_structure']
        attach_cycles(g, cycle_len=k, count=count)

        add_to_list(all_graphs, g, label)

    label += 1

    for i in range(sample_size):
        a, b = distrib_1['dims']
        g = random_tree(random.randint(a, b))

        a, b = distrib_1['count']
        count = random.randint(1, 2)

        k = distrib_0['len_structure']
        attach_cycles(g, cycle_len=k, count=count, is_clique=True)

        add_to_list(all_graphs, g, label)

    label += 1
    return all_graphs


def write_gexf(output_path: Path, graphs):
    print('Created .gexf files in %s' % output_path)
    for g, label in graphs:
        nx.write_gexf(
            g,
            output_path / ('%d.%d.gexf' % (g.graph['graph_num'], label))
        )


def write_adjacency(output_path: Path, dataset: str, graphs):
    relabled_gs = []
    first_label = 1
    graph_indicator = []
    for g, label in tq(graphs):
        relabled_gs.append(nx.convert_node_labels_to_integers(
            g,
            first_label=first_label))
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
