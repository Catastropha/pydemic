# import torch
from . import factories
# import time


def test_population():
    population = factories.PopulationFactory.create()
    pso_agent = factories.PSOAgentFactory.create()
    gwo_agent = factories.GWOAgentFactory.create()
    pso_swarm = factories.SwarmFactory.create()
    gwo_swarm = factories.SwarmFactory.create(memory_size=3)

    pso_swarm.populate(agent=pso_agent, n_agents=2)
    gwo_swarm.populate(agent=gwo_agent, n_agents=2)

    # data = torch.tensor([
    #     [3.0, 3.0, 3.0],
    #     [1.0, 1.0, 1.0],
    #     [2.0, 2.0, 2.0],
    # ])

    population.add(pso_swarm)
    if not len(population.swarms) == 1:
        raise AssertionError()
    if not population.swarms[0] == pso_swarm:
        raise AssertionError()
    population.add(gwo_swarm)
    if not len(population.swarms) == 2:
        raise AssertionError()
    if not population.swarms[1] == gwo_swarm:
        raise AssertionError()

    # population.run(data=data, episodes=10)
    # episode, episodes = pso_swarm.when()
    #
    # if not episode == 10:
    #     raise AssertionError()
    # episode, episodes = gwo_swarm.when()
    # if not episode == 10:
    #     raise AssertionError()
