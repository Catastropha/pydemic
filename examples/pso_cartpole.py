import gym
import torch
import torch.nn as nn

from pydemic.algos import PSO


env = gym.make('CartPole-v1')
action_space = env.action_space.n
space_size = env.observation_space.shape[0]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(space_size, 4)
        self.fc2 = nn.Linear(4, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


pso = PSO(
    public_coefficient=1.5,
    private_coefficient=1.5,
    inertia=0.2,
)

pso.learn(
    device='cpu',
    env=env,
    mode='discrete',
    score_threshold=190,
    model=Model,
    n_agents=8,
    epochs=500,
    filename='best_agent',
    verbose=True,
)
