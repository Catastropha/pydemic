from abc import ABCMeta, abstractmethod
import torch
from typing import Tuple
from .distributions import levy
from .memories import Memory


class BaseAgent(metaclass=ABCMeta):

    def __init__(self,
                 n_clusters: int,
                 memory_size: int = 1,
                 ):
        self.memory = Memory(memory_size=memory_size)
        self.n_clusters = n_clusters
        self.centroids = None
        self.swarm = None

    @abstractmethod
    def train(self):
        pass

    def initialize(self,
                   data: torch.Tensor,
                   swarm,
                   ) -> None:
        indexes = torch.randint(0, data.shape[0], (self.n_clusters,))
        self.centroids = data[indexes].clone()
        self.memorize(score=float('inf'), position=self.centroids)
        self.swarm = swarm

    def memorize(self,
                 score: float,
                 position: torch.Tensor,
                 ) -> None:
        self.memory.add(score=score, obj=position.clone())

    def topk(self,
             k: int,
             ) -> list:
        return self.memory.topk(k=k)

    def run(self,
            data: torch.Tensor,
            ) -> Tuple[torch.Tensor, float]:

        distances = []
        for centroid in self.centroids:
            distance = torch.pow(data - centroid, 2).sum(1)
            distances.append(distance)
        distances = torch.stack(distances, dim=1)
        prediction = torch.argmin(distances, dim=1)

        score = 0.0
        for i, centroid in enumerate(self.centroids):
            indexes = (prediction == i).nonzero()
            score += torch.pow(data[indexes] - centroid, 2).sum()

        self.memorize(score=score, position=self.centroids)

        return prediction, score


class PSOAgent(BaseAgent):

    def __init__(self,
                 n_clusters: int,
                 social_coefficient: float,
                 personal_coefficient: float,
                 inertia: float,
                 ):
        BaseAgent.__init__(self, n_clusters=n_clusters, )
        self.social_coefficient = social_coefficient
        self.personal_coefficient = personal_coefficient
        self.inertia = inertia
        self.velocity = 0

    def train(self) -> None:
        social_best_position = self.swarm.topk(k=1)[0].topk(k=1)[0]
        personal_best_position = self.topk(k=1)[0]
        inertia = self.inertia * self.velocity
        personal = self.personal_coefficient * torch.empty(self.centroids.shape).uniform_(0, 1) * (personal_best_position - self.centroids)
        social = self.social_coefficient * torch.empty(self.centroids.shape).uniform_(0, 1) * (social_best_position - self.centroids)
        self.velocity = inertia + personal + social
        self.centroids += self.velocity


class CSAgent(BaseAgent):

    def __init__(self,
                 n_clusters: int,
                 levy_alpha: int,
                 expose: int,
                 discoverability: float,
                 ):
        BaseAgent.__init__(self,  n_clusters=n_clusters, )
        self.levy_alpha = levy_alpha
        self.expose = expose
        self.discoverability = discoverability

    def train(self) -> None:
        social_best_position = self.swarm.topk(k=1)[0].topk(k=1)[0]
        change = 2 * levy(shape=self.centroids.shape, alpha=self.levy_alpha) * (social_best_position - self.centroids)
        self.centroids += change

        if self in self.swarm.bottomk(k=self.expose):
            rand = torch.empty(1).uniform_(0, 1)
            if (rand < self.discoverability)[0]:
                agents = self.swarm.agent_sample(2)
                rand = torch.empty(1).uniform_(0, 1)
                change = rand * (agents[0].centroids - agents[1].centroids)
                self.centroids += change


class GWOAgent(BaseAgent):

    def __init__(self,
                 n_clusters: int,
                 ):
        BaseAgent.__init__(self,  n_clusters=n_clusters, )

    def train(self) -> None:
        episode, episodes = self.swarm.when()
        a = 2 - 2 * episode / episodes

        alpha, beta, delta = self.swarm.topk(k=3)
        alpha = alpha.topk(1)[0]
        beta = beta.topk(1)[0]
        delta = delta.topk(1)[0]

        aa_alpha = 2 * a * torch.empty(self.centroids.shape).uniform_(0, 1) - a
        aa_beta = 2 * a * torch.empty(self.centroids.shape).uniform_(0, 1) - a
        aa_delta = 2 * a * torch.empty(self.centroids.shape).uniform_(0, 1) - a

        cc_alpha = 2 * torch.empty(self.centroids.shape).uniform_(0, 1)
        cc_beta = 2 * torch.empty(self.centroids.shape).uniform_(0, 1)
        cc_delta = 2 * torch.empty(self.centroids.shape).uniform_(0, 1)

        dd_alpha = abs(cc_alpha * alpha - self.centroids)
        dd_beta = abs(cc_beta * beta - self.centroids)
        dd_delta = abs(cc_delta * delta - self.centroids)

        x_alpha = alpha - aa_alpha * dd_alpha
        x_beta = beta - aa_beta * dd_beta
        x_delta = delta - aa_delta * dd_delta

        self.centroids = (x_alpha + x_beta + x_delta) / 3


class FPAgent(BaseAgent):

    def __init__(self,
                 n_clusters: int,
                 levy_alpha: int,
                 switch_probability: float,
                 ):
        BaseAgent.__init__(self,  n_clusters=n_clusters, )
        self.levy_alpha = levy_alpha
        self.switch_probability = switch_probability

    def train(self) -> None:
        switch = torch.empty(1).uniform_(0, 1)
        if switch < self.switch_probability:
            social_best_position = self.swarm.topk(k=1)[0].topk(k=1)[0]
            self.centroids += levy(shape=self.centroids.shape, alpha=self.levy_alpha) * (social_best_position - self.centroids)
        else:
            agents = self.swarm.agent_sample(2)
            self.centroids += torch.empty(self.centroids.shape).uniform_(0, 1) * (agents[0].centroids - agents[1].centroids)

