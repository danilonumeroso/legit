# LEGIT: Local Explainer for deep Graph networks by Input perturbaTion
This repository contains the official implementation of LEGIT.

# Usage
We assume miniconda (or anaconda) to be installed.

First, install the dependencies:
```
source setup/install.sh [cpu | cu92 | cu101 | cu102]
```
By default, the script will install the 1.6.0 cpu version of PyTorch and PyTorch Geometric. If you want to install the cuda version, just pass the argument. Instead, if you wish to install a different version (e.g, 1.7+) you need to modify the first line of the script:

```
#!/bin/sh
CUDA_VERSION=${1:-cpu}
TORCH_VERSION=1.6.0 # modify this
TORCH_GEOMETRIC_VERSION=1.6.0 # and this
```

The setup script will create a conda environment named "meg".

Now you can train the DGN to be explained by running:
```
python train_dgn.py [tox21 | esol] <experiment_name>
```

Executing this script will populate the directory `runs`, which is structured as follows:

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

## Reproducibility

To reproduce results comparable to what is shown in the paper, run:
```
python train_dgn.py [tox21 | esol | cycliq] <experiment_name> --lr LR --hidden-size HS  --batch-size BS --dropout D --epochs 100

python train_gen.py [tox21 | esol | cycliq] <experiment_name>
```

The generator will dump neighbours to `runs/<dataset_name>/<experiment_name>/gen_output/<sample>/data.json`.

To generate explanation for a sample, run:

```
python explain.py contrast | GNNExplainer | linear | random <...parameters>
```

`contrast` only work for CYCLIQ, whereas `linear` requires chemistry tasks (i.e, TOX21, ESOL).

To evaluate explanation accuracy for CYCLIQ, we use the evaluation script from (CoGE)[https://github.com/lukasjf/contrastive-gnn-explanation]:

```
python evaluate.py <dataset_path> <explain_path>
```
