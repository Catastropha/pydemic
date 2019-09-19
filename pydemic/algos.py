from abc import ABCMeta, abstractmethod
from collections import deque
import numpy as np
import torch
from .memories import Memory
from .agents import PSOAgent


class BaseAlgo(metaclass=ABCMeta):

    def __init__(self,
                 public_memory_size: int = 1,
                 ):
        self.public_memory = Memory(memory_size=public_memory_size)
        self.agents = []

    @abstractmethod
    def populate(self, device, model, n_agents):
        pass

    def save(self, filename):
        agent = self.public_memory.topk(k=1)[0]
        torch.save(obj=agent.model.state_dict(), f=filename)

    def learn(self,
              device,
              env,
              mode,
              score_threshold,
              model,
              n_agents,
              epochs,
              filename='best_agent',
              verbose=False,
              ):

        self.populate(device=device, model=model, n_agents=n_agents)

        all_scores = []
        scores_window = deque(maxlen=100)

        for epoch in range(1, epochs+1):

            scores = []
            for agent in self.agents:

                state = env.reset()
                score = 0

                while True:
                    action = agent.act(state)
                    if mode == 'discrete':
                        action = np.argmax(action)
                    next_state, reward, done, _ = env.step(action)

                    score += reward
                    state = next_state

                    if done:
                        break

                self.public_memory.add(score=score, obj=agent)
                scores.append(score)

            avg_score = np.mean(scores)
            scores_window.append(avg_score)
            all_scores.append(avg_score)

            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)), end="")
                if epoch % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)))
            if np.mean(scores_window) >= score_threshold:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)))
                self.save(filename=filename)
                break

        return all_scores


class PSO(BaseAlgo):
    def __init__(self,
                 public_coefficient: float,
                 private_coefficient: float,
                 inertia: float,
                 ):
        BaseAlgo.__init__(self, )
        self.public_coefficient = public_coefficient
        self.private_coefficient = private_coefficient
        self.inertia = inertia

    def populate(self, device, model, n_agents):
        self.agents = []
        for _ in range(n_agents):
            private_memory = Memory()
            agent = PSOAgent(
                device=device,
                model=model(),
                public_memory=self.public_memory,
                private_memory=private_memory,
                public_coefficient=self.public_coefficient,
                private_coefficient=self.private_coefficient,
                inertia=self.inertia,
            )
            self.agents.append(agent)
