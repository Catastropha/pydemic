# Gradient free reinforcement learning for PyTorch

[![Version](https://img.shields.io/pypi/v/pydemic.svg?style=flat)](https://pypi.org/project/pydemic/#history)
[![Downloads](https://pepy.tech/badge/pydemic)](https://pepy.tech/project/pydemic)
![License](https://img.shields.io/pypi/l/pyriad.svg?style=flat)

`pydemic` offers reinforcement learning algorithms built with Python on top of the deep learning library [PyTorch](https://pytorch.org/).

You can extend `pydemic` according to your own needs. You can implement custom algorithms by extending simple abstract classes.

## Algorithms
As of today, the following algorithms have been implemented:

-   [x] Particle Swarm Optimization (PSO) [[1]](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)
-   [x] Grey Wolf Optimization (GWO) [[3]](https://www.researchgate.net/profile/Mohammed_Bakr6/post/how_to_implement_Open_Vechile_Routing_Problem_using_Grey_Wolf_Optimizer/attachment/59d621c66cda7b8083a1b3fa/AS%3A273784001499151%401442286600462/download/GWO_finalVersion.pdf)

## Installation

1.  Install PyTorch. You can find it here: [PyTorch](https://pytorch.org/)
2.  `pip install pydemic`

## Examples

You can find examples in `examples/` directory

You can also run examples: `python examples/pso_cartpole.py`

You might want to `export PYTHONPATH=/path/to/this/directory`

## Contribute

1.  Implement new algorithms
2.  Improve code design
3.  Improve comments and readme
4.  Tests
