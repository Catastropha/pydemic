from abc import ABCMeta, abstractmethod
from collections import deque
from .memories import Memory
from .agents import PSOAgent


class BaseAlgo(metaclass=ABCMeta):

    def __init__(self, ):
        self.agents = []

    @abstractmethod
    def populate(self, model, n_agents):
        pass

    def learn(self,
              env,
              score_threshold,
              model,
              n_agents,
              epochs,
              verbose=False,
              ):

        self.populate(model=model, n_agents=n_agents)

        all_scores = []
        scores_window = deque(maxlen=100)

        for epoch in range(1, epochs+1):

            for agent in self.agents:

                state = env.reset()
                score = 0

                while True:
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)

                    score += reward
                    state = next_state

                    if done:
                        break

                avg_score = np.mean(score)
                scores_window.append(avg_score)
                all_scores.append(avg_score)

                print('\rEpisode {}\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)), end="")
                if epoch % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)))
                if np.mean(scores_window) >= score_threshold:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(epoch, np.mean(scores_window)))
                    self.save(filename=filename)
                    break


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

    def populate(self, model, n_agents):
        self.agents = []
        public_memory = Memory()
        for _ in range(n_agents):
            private_memory = Memory()
            agent = PSOAgent(
                model=model(),
                public_memory=public_memory,
                private_memory=private_memory,
                public_coefficient=self.public_coefficient,
                private_coefficient=self.private_coefficient,
                inertia=self.inertia,
            )
            self.agents.append(agent)




