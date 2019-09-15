import torch
from . import factories


def test_base_agent_initialize():
    agent = factories.PSOAgentFactory.create()
    swarm = factories.SwarmFactory.create()

    data = torch.tensor([
        [3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    agent.initialize(data=data, swarm=swarm)

    if not agent.swarm == swarm:
        raise AssertionError()
    if not len(agent.centroids) == agent.n_clusters:
        raise AssertionError()


def test_base_agent_memorize():
    agent = factories.PSOAgentFactory.create()

    tensor = torch.tensor([3.0, 3.0, 3.0])
    agent.memorize(score=1, position=tensor)

    if not torch.all(torch.eq(tensor, agent.memory.memory[0][1])) == True:
        raise AssertionError()


def test_base_agent_topk():
    agent = factories.PSOAgentFactory.create()

    tensor = torch.tensor([3.0, 3.0, 3.0])
    agent.memorize(score=1, position=tensor)
    topk = agent.topk(k=1)[0]

    if not torch.all(torch.eq(tensor, topk)) == True:
        raise AssertionError()


