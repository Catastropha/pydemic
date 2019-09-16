import gym
import torch
import torch.nn as nn

from pydemic import PSO


env = gym.make('CartPole-v1')
action_space = env.action_space.n
space_size = env.observation_space.shape[0]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = Model()

pso = PSO()
pso.learn(env=env, epochs=500, verbose=True)