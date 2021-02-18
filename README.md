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

...
