from collections import deque
from typing import Tuple
import copy
import torch
import random
from .memories import Memory


class Swarm:

    def __init__(self,
                 memory_size: int = 1,
                 ):
        self.memory = Memory(memory_size=memory_size)
        self.agents = []
        self.episode = None
        self.episodes = None

    def populate(self,
                 agent,
                 n_agents: int,
                 ) -> None:
        self.agents.append(agent)
        for _ in range(n_agents - 1):
            clone = copy.deepcopy(agent)
            self.agents.append(clone)

    def initialize(self,
                   data: torch.Tensor,
                   ) -> None:
        for agent in self.agents:
            agent.initialize(data=data, swarm=self)
            self.memorize(score=float('inf'), agent=agent)

    def memorize(self,
                 score: float,
                 agent,
                 ) -> None:
        self.memory.add(score=score, obj=agent)

    def topk(self,
             k: int,
             ) -> list:
        return self.memory.topk(k=k)

    def bottomk(self,
             k: int,
             ) -> list:
        return self.memory.bottomk(k=k)

    def agent_sample(self,
                     n: int,
                     ) -> list:
        return random.sample(self.agents, n)

    def when(self) -> Tuple[int, int]:
        return self.episode, self.episodes

    def predict(self,
                data: torch.Tensor,
                ) -> Tuple[list, float]:
        agent = self.agents[0]
        labels, score = agent.run(data=data)
        return labels, score

    def train(self,
              data: torch.Tensor,
              episodes: int,
              ) -> None:
        self.episodes = episodes
        all_scores = []
        scores_window = deque(maxlen=50)

        self.initialize(data=data)

        for self.episode in range(1, episodes + 1):
            scores = []

            for agent in self.agents:
                _, score = agent.run(data)
                self.memorize(score=score, agent=agent)
                agent.train()
                scores.append(score)

            avg_score = torch.stack(scores).mean()
            scores_window.append(avg_score)
            all_scores.append(avg_score)
            scores_window_mean = torch.stack(list(scores_window)).mean()

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, scores_window_mean), end="")
            if self.episode % 50 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.episode, scores_window_mean))


