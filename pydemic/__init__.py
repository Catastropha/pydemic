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
        public_memory = Memory()
        for _ in range(n_agents):
            private_memory = Memory()
            agents.append(PSOAgent(model=model, public_memory=public_memory, private_memory=private_memory))

    elif algo == 'GWO':
        public_memory = Memory(memory_size=3)
        for _ in range(n_agents):
            private_memory = Memory()
            agents.append(GWOAgent(model=model, public_memory=public_memory, private_memory=private_memory))
