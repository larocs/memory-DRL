# Memory Deep Reinforcement Learning

Memory-based implementation of [Soft Actor Critic](https://arxiv.org/abs/1801.01290) for humanoid locomotion tasks.

## Installation

### CoppeliaSim

Our experiments uses the EDU version of CoppeliaSim as the simulation platform. You can install it with no costs following [these instructions](https://www.coppeliarobotics.com/downloads).

We also use PyRep as the Python API with Coppelia, which can be installed [from their GitHub](https://github.com/stepjam/PyRep).

### This Project
The code was developed in Linux, but should work fine in Windows platforms.
We recomend using Python virtual environments to isolate this project dependencies from the host system dependencies.

This project is implemented as a Python package, and therefore should be installed as a dependency:

```shell
python -m venv mem_sac
source mem_sac/bin/activate

git clone git@github.com:larocs/memory-DRL.git
pip install -e memory-DRL
```

## Usage

We use a Json file to describe experiments (model architectures, environemnt parameters and etc) in order to facilitate experiment versioning through git. You can find some examples at `examples` folder. 

Training a model can be done through an auxiliar script in this project (preferably copy this script to another folder):

```shell
python mysac/runner.py --exp_path examples/cartpole_fcn
```

And to execute the trained policy:
```shell
python mysac/run_policy.py --exp_path examples/cartpole_fcn
```

Training statistics can be found in the experiment folder, along with the trained policy:
```shell
ls examples/cartpole_fcn/stats/
```

We also include an auxiliary scripts for quick plotting these results:
```shell
# Training evaluation results
python mysac/plot.py examples/cartpole_fcn/stats/eval_stats.csv

# SAC statistics during training
python mysac/plot_sac_stats.py --exp_path examples/cartpole_fcn/
```

### Framework

Since the code is implemented as a framework, all models and environment are modular and can be freely extended to allow new experiments. Most of the code is documented through Python docstrings. 
## Authors

Esther Luna Colombini & Samuel Chenatti