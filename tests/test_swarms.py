import torch
from . import factories


def test_swarm_populate():
    agent = factories.PSOAgentFactory.create()
    swarm = factories.SwarmFactory.create()

    swarm.populate(agent=agent, n_agents=2)

    if not len(swarm.agents) == 2:
        raise AssertionError()

    if not type(swarm.agents[0]) == type(swarm.agents[1]):
        raise AssertionError()


def test_swarm_memorize_topk_bottomk_sample():
    agent = factories.PSOAgentFactory.create()
    swarm = factories.SwarmFactory.create()

    data = torch.tensor([
        [3.0, 3.0, 3.0],
    ])

    swarm.populate(agent=agent, n_agents=2)
    swarm.initialize(data=data)

    swarm.memorize(score=1, agent=swarm.agents[0])
    topk = swarm.topk(k=1)
    bottomk = swarm.bottomk(k=1)
    agents = swarm.agent_sample(n=1)
    agents2 = swarm.agent_sample(n=2)

    if not torch.all(torch.eq(swarm.agents[0].centroids, data[0])) == True:
        raise AssertionError()
    if not len(swarm.memory.memory) == 1:
        raise AssertionError()
    if not len(topk) == 1:
        raise AssertionError()
    if not topk[0] == swarm.agents[0]:
        raise AssertionError()
    if not len(bottomk) == 1:
        raise AssertionError()
    if not bottomk[0] == swarm.agents[0]:
        raise AssertionError()
    if not len(agents) == 1:
        raise AssertionError()
    if not agents[0] in swarm.agents:
        raise AssertionError()
    if not len(agents2) == 2:
        raise AssertionError()
    if not agents2[0] in swarm.agents or not agents2[1] in swarm.agents:
        raise AssertionError()


def test_swarm_when():
    swarm = factories.SwarmFactory.create()
    episode, episodes = swarm.when()

    if not episode == None or not episodes == None:
        raise AssertionError()
