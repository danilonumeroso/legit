# LEGIT: Local Explainer for deep Graph networks by Input perturbaTion
This repository contains the official implementation of LEGIT.

## Install dependencies
```
    conda env create -f envs/cpu.yml
```
or
```
    conda env create -f envs/gpu.yml
```

## Train the predictor
Now you can train the graph network (with default hyperparameters) to be explained by running:
```
python dgn.py train [tox21 | esol | cycliq] <experiment_name>
```
If you wish to tune the hyperparameters, refer to the `model-selection` file.

Executing this command  will populate the directory `runs`, which is structured as follows:
```
runs/
└── <dataset_name>
    └── <experiment_name>
        ├── best_performance.json
        ├── ckpt
        │   └── <model>.pth
        ├── hyperparams.json
        ├── gen_output
        │   └── <sample>
        │       └── data.json
        │
        ├── plots
        │   └── events.out.tfevents.*
        └── splits
            ├── test.pth
            ├── train.pth
            └── val.pth

```

## Get graph embeddings
If you wish to replicate results of (some of) the baselines, you'll be required to have the node embeddings of the training and test graphs, especially for CoGE.
You can get them by running:
```
python dgn.py embed [tox21 | esol | cycliq] <experiment_name>
```
Embeddings will be saved to `runs/<dataset_name>/<experiment_name>/embeddings/[train | test]`.

## Train the generator
After the DGN has been trained, run:
```
python train_gen.py [tox21 | esol | cycliq] <experiment_name>
```
The generator will start searching for the best neighbours and will dump them to `runs/<dataset_name>/<experiment_name>/gen_output/<sample>/data.json`.

## Generate explanations
Once we have the neighbours for each test sample you would like to explain, run:

```
python explain.py contrast | linear  <...parameters>
```

`contrast` only work for CYCLIQ, whereas `linear` requires chemistry tasks (i.e, TOX21, ESOL).

To evaluate explanation accuracy for CYCLIQ, we use the evaluation script from (CoGE)[https://github.com/lukasjf/contrastive-gnn-explanation]:

```
python evaluate.py <dataset_path> <explain_path>
```

For more information on the available explainers, run `python explain.py --help` or `python explain.py <method> --help`.
