from abc import ABCMeta, abstractmethod
import torch
from .memories import Memory


class BaseAgent(metaclass=ABCMeta):

    def __init__(self,
                 device,
                 model,
                 ):
        self.device = device
        self.model = model

    @abstractmethod
    def train(self):
        pass

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            return self.model(state).cpu().data.numpy()


class PSOAgent(BaseAgent):

    def __init__(self,
                 device,
                 model,
                 public_memory: Memory,
                 private_memory: Memory,
                 public_coefficient: float,
                 private_coefficient: float,
                 inertia: float,
                 ):
        BaseAgent.__init__(self, device=device, model=model)
        self.public_memory = public_memory
        self.private_memory = private_memory
        self.public_coefficient = public_coefficient
        self.private_coefficient = private_coefficient
        self.inertia = inertia
        self.velocity = 0

    def train(self) -> None:
        public_best_position = self.public_memory.topk(k=1)[0].topk(k=1)[0]
        private_best_position = self.private_memory.topk(k=1)[0]
        inertia = self.inertia * self.velocity
        private = self.private_coefficient * torch.empty(self.centroids.shape).uniform_(0, 1) * (private_best_position - self.centroids)
        public = self.public_coefficient * torch.empty(self.centroids.shape).uniform_(0, 1) * (public_best_position - self.centroids)
        self.velocity = inertia + private + public
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
