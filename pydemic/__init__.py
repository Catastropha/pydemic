import torch.nn as nn
from .agents import PSOAgent, GWOAgent
from .memories import Memory


def learn(env,
          model,
          algo: str,
          n_agents: int,
          epochs: int,
          verbose: bool = True,
          ) -> None:

    agents = []

    if algo == 'PSO':
        global_memory = Memory()
        agents = [PSOAgent() for _ in range(n_agents)]

    elif algo == 'GWO':
        global_memory = Memory(memory_size=3)
        agents = [GWOAgent() for _ in range(n_agents)]