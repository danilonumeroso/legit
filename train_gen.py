import torch
import numpy as np
import json
import os
import networkx as nx
import typer

from models.explainer import CF_Tox21, NCF_Tox21, Agent, CF_Esol, NCF_Esol, CF_Cycliq, NCF_Cycliq
from torch.utils.tensorboard import SummaryWriter
from utils import SortedQueue, morgan_bit_fingerprint, get_split, get_dgn, set_seed, mol_to_smiles, x_map_tox21, pyg_to_mol_tox21, mol_from_smiles
from torch.nn import functional as F
from torch_geometric.utils import to_networkx


def tox21(general_params, base_path, writer, num_counterfactuals,
          num_non_counterfactuals, original_molecule, model_to_explain,
          **args):

    out, (_,
          original_encoding) = model_to_explain(original_molecule.x,
                                                original_molecule.edge_index)

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    assert pred_class == original_molecule.y.item()

    original_molecule.smiles = mol_to_smiles(
        pyg_to_mol_tox21(original_molecule))

    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = [
        x_map_tox21(e).name for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()])
    ]

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'combined'
    }

    cf_queue = SortedQueue(num_counterfactuals,
                           sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Tox21(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(num_non_counterfactuals,
                            sort_predicate=lambda mol: mol['reward'])
    ncf_env = NCF_Tox21(**params)
    ncf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'],
                                      args['fp_radius']).numpy()

    gen_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker=lambda x: "cf" if x != pred_class else "ncf",
              tb_name="tox21",
              id_function=lambda action: action,
              args=args)
    gen_train(writer,
              action_encoder,
              args['fp_length'],
              ncf_env,
              ncf_queue,
              marker=lambda x: "cf" if x != pred_class else "ncf",
              tb_name="tox_21",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'bin_classification',
            'output': logits.numpy().tolist(),
            'for_explanation': original_molecule.y.item(),
            'class': original_molecule.y.item()
        }
    })
    overall_queue.extend(cf_queue.data_)
    overall_queue.extend(ncf_queue.data_)

    save_results(base_path, overall_queue, args)


def cycliq(general_params, base_path, writer, num_counterfactuals,
           num_non_counterfactuals, original_graph, model_to_explain, **args):

    out, (node_embs,
          original_encoding) = model_to_explain(original_graph.x,
                                                original_graph.edge_index)

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    assert pred_class == original_graph.y.item()

    params = {
        'init_graph': original_graph,
        'allow_removal': general_params['allow_removal'],
        'allow_node_addition': general_params['allow_node_addition'],
        'allow_edge_addition': general_params['allow_edge_addition'],
        'allow_no_modification': general_params['allow_no_modification'],
        'discount_factor': general_params['discount_factor'],
        'max_steps': general_params['max_steps'],
        # Task-specific params
        'original_graph': original_graph,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.4,
        'similarity_measure': 'neural_encoding'
    }

    cf_queue = SortedQueue(num_counterfactuals,
                           sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Cycliq(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(num_non_counterfactuals,
                            sort_predicate=lambda mol: mol['reward'])
    params['max_steps'] = 1
    params['weight_sim'] = 0.6
    params['allow_edge_addition'] = True
    params['allow_removal'] = False
    ncf_env = NCF_Cycliq(**params)
    ncf_env.initialize()

    def action_encoder(action):
        return model_to_explain(action.x, action.edge_index)[1][1].numpy()

    try:
        gen_train(
            writer,
            action_encoder,
            model_to_explain.num_hidden * 2,
            cf_env,
            cf_queue,
            marker=lambda x: "cf" if x != pred_class else "ncf",
            tb_name="cycliq",
            id_function=lambda action: hash(map(tuple, action.edge_index)),
            args=args)
    except KeyboardInterrupt:
        print("Cycle interrupted.")

    try:
        args['epochs'] = 100
        gen_train(
            writer,
            action_encoder,
            model_to_explain.num_hidden * 2,
            ncf_env,
            ncf_queue,
            marker=lambda x: "cf" if x != pred_class else "ncf",
            tb_name="cycliq",
            id_function=lambda action: hash(map(tuple, action.edge_index)),
            args=args)
    except KeyboardInterrupt:
        print("Cycle interrupted.")

    print("Save")
    overall_queue = []
    overall_queue.append({
        'pyg': original_graph,
        'marker': 'og',
        'encoding': node_embs.numpy(),
        'prediction': {
            'type': 'bin_classification',
            'output': logits.numpy().tolist(),
            'for_explanation': original_graph.y.item(),
            'class': original_graph.y.item()
        }
    })

    # ensure that num_cf == num_non_cf (for cycliq only)
    N = len(list(filter(lambda x: x['marker'] == 'cf',  cf_queue.data_)))

    overall_queue.extend(cf_queue.data_[:N])
    overall_queue.extend(ncf_queue.data_[:N])

    save_results(base_path, overall_queue, args, quantitative=True)


def esol(general_params, base_path, writer, num_counterfactuals,
         num_non_counterfactuals, original_molecule, model_to_explain, **args):
    original_molecule.x = original_molecule.x.float()

    og_prediction, original_encoding = model_to_explain(
        original_molecule.x, original_molecule.edge_index)
    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = np.unique([
        x.GetSymbol()
        for x in mol_from_smiles(original_molecule.smiles).GetAtoms()
    ])

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'model_to_explain': model_to_explain,
        'original_molecule': original_molecule,
        'weight_sim': 0.2,
        'similarity_measure': 'combined',
    }

    cf_queue = SortedQueue(num_counterfactuals,
                           sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Esol(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(num_non_counterfactuals,
                            sort_predicate=lambda mol: mol['reward'])
    ncf_env = NCF_Esol(**params)
    ncf_env.initialize()

    def action_encoder(action):
        return morgan_bit_fingerprint(action, args['fp_length'],
                                      args['fp_radius']).numpy()

    gen_train(writer,
              action_encoder,
              args['fp_length'],
              cf_env,
              cf_queue,
              marker=lambda _: "cf",
              tb_name="esol",
              id_function=lambda action: action,
              args=args)
    gen_train(writer,
              action_encoder,
              args['fp_length'],
              ncf_env,
              ncf_queue,
              marker=lambda _: "ncf",
              tb_name="esol",
              id_function=lambda action: action,
              args=args)

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': og_prediction.squeeze().detach().numpy().tolist(),
            'for_explanation':
            og_prediction.squeeze().detach().numpy().tolist()
        }
    })
    overall_queue.extend(cf_queue.data_)
    overall_queue.extend(ncf_queue.data_)

    save_results(base_path, overall_queue, args)


def gen_train(writer, action_encoder, n_input, environment, queue, marker,
              tb_name, id_function, args):
    device = torch.device("cpu")
    agent = Agent(n_input + 1, 1, device, args['lr'],
                  args['replay_buffer_size'])

    eps = 1.0
    batch_losses = []
    episode = 0
    it = 0

    while episode < args['epochs']:
        steps_left = args['max_steps_per_episode'] - environment.num_steps_taken
        valid_actions = list(environment.get_valid_actions())

        observations = np.vstack([
            np.append(action_encoder(action), steps_left)
            for action in valid_actions
        ])

        observations = torch.as_tensor(observations).float()
        a = agent.action_step(observations, eps)
        action = valid_actions[a]

        result = environment.step(action)

        action_embedding = np.append(action_encoder(action), steps_left)

        _, out, done = result

        writer.add_scalar(f'{tb_name}/reward', out['reward'], it)
        writer.add_scalar(f'{tb_name}/prediction', out['reward_pred'], it)
        writer.add_scalar(f'{tb_name}/similarity', out['reward_sim'], it)

        steps_left = args['max_steps_per_episode'] - environment.num_steps_taken

        action_embeddings = np.vstack([
            np.append(action_encoder(action), steps_left)
            for action in environment.get_valid_actions()
        ])

        agent.replay_buffer.push(
            torch.as_tensor(action_embedding).float(),
            torch.as_tensor(out['reward']).float(),
            torch.as_tensor(action_embeddings).float(),
            float(result.terminated))

        if it % args['update_interval'] == 0 and len(
                agent.replay_buffer) >= args['batch_size']:
            loss = agent.train_step(args['batch_size'], args['gamma'],
                                    args['polyak'])
            loss = loss.item()
            batch_losses.append(loss)

        it += 1

        if done:
            episode += 1

            print(
                f'({args["sample"]}) Episode {episode}> Reward = {out["reward"]:.4f} (pred: {out["reward_pred"]:.4f}, sim: {out["reward_sim"]:.4f})'
            )

            pred_class = out['prediction']['class']
            queue.insert({'marker': marker(pred_class), 'id': id_function(action), **out})

            eps *= 0.9987
            # eps = max(eps, 0.05)

            batch_losses = []
            environment.initialize()


def save_results(base_path, queue, args, quantitative=False):
    output_dir = base_path + f"/gen_output/{args['sample']}"
    embedding_dir = output_dir + "/embeddings"
    gexf_dir = output_dir + "/gexf_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(embedding_dir)
        os.makedirs(gexf_dir)

    for i, molecule in enumerate(queue):
        np.save(embedding_dir + f"/{i}", molecule.pop('encoding'))
        pyg = molecule.pop('pyg')
        if quantitative:
            g = to_networkx(pyg, to_undirected=True)
            nx.write_gexf(
                g, f"{gexf_dir}/{i}.{molecule['prediction']['class']}.gexf")

    with open(output_dir + "/data.json", "w") as outf:
        json.dump(queue, outf, indent=2)


def main(dataset: str,
         experiment_name: str = typer.Argument("test"),
         sample: int = typer.Option(0),
         epochs: int = typer.Option(5000),
         max_steps_per_episode: int = typer.Option(1),
         num_counterfactuals: int = typer.Option(10),
         num_non_counterfactuals: int = typer.Option(10),
         fp_length: int = typer.Option(1024),
         fp_radius: int = typer.Option(2),
         lr: float = typer.Option(1e-4),
         polyak: float = typer.Option(0.995),
         gamma: float = typer.Option(0.95),
         discount: float = typer.Option(0.9),
         replay_buffer_size: int = typer.Option(10000),
         batch_size: int = typer.Option(1),
         update_interval: int = typer.Option(1),
         allow_no_modification: bool = typer.Option(False),
         allow_removal: bool = typer.Option(True),
         allow_node_addition: bool = typer.Option(True),
         allow_edge_addition: bool = typer.Option(True),
         allow_bonds_between_rings: bool = typer.Option(True),
         seed: int = typer.Option(0)):

    general_params = {
        # General-purpose params
        'discount_factor': discount,
        'allow_removal': allow_removal,
        'allow_no_modification': allow_no_modification,
        'allow_bonds_between_rings': allow_bonds_between_rings,
        'allow_node_addition': allow_node_addition,
        'allow_edge_addition': allow_edge_addition,
        'allowed_ring_sizes': set([5, 6]),
        'max_steps': max_steps_per_episode,
        'fp_len': fp_length,
        'fp_rad': fp_radius
    }

    dataset = dataset.lower()
    if dataset == 'tox21':
        gen = tox21
    elif dataset == 'esol':
        gen = esol
    elif dataset == 'cycliq':
        gen = cycliq

    set_seed(seed)

    base_path = f'./runs/{dataset.lower()}/{experiment_name}'

    gen(general_params,
        base_path,
        SummaryWriter(f'{base_path}/plots'),
        num_counterfactuals,
        num_non_counterfactuals,
        get_split(dataset.lower(), 'test', experiment_name)[sample],
        model_to_explain=get_dgn(dataset.lower(), experiment_name),
        experiment_name=experiment_name,
        sample=sample,
        epochs=epochs,
        max_steps_per_episode=max_steps_per_episode,
        fp_length=fp_length,
        fp_radius=fp_radius,
        lr=lr,
        polyak=polyak,
        gamma=gamma,
        discount=discount,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        update_interval=update_interval)


if __name__ == '__main__':
    typer.run(main)
