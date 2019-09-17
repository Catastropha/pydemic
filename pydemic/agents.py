from abc import ABCMeta, abstractmethod
import torch


class BaseAgent(metaclass=ABCMeta):

    def __init__(self,
                 model,
                 ):
        self.model = model

    @abstractmethod
    def train(self):
        pass

    def initialize(self,
                   data: torch.Tensor,
                   swarm,
                   ) -> None:
        # todo
        self.memorize(score=float('inf'), model=self.model)

    def memorize(self,
                 score: float,
                 model,
                 ) -> None:
        self.memory.add(score=score, obj=model.clone())

    def topk(self,
             k: int,
             ) -> list:
        return self.memory.topk(k=k)


class PSOAgent(BaseAgent):

    def __init__(self,
                 social_coefficient: float,
                 personal_coefficient: float,
                 inertia: float,
                 ):
        BaseAgent.__init__(self, )
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


class GWOAgent(BaseAgent):

    def __init__(self, ):
        BaseAgent.__init__(self, )

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
