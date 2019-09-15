import factory
from pyriad import memories
from pyriad import agents
from pyriad import swarms
from pyriad import populations


class MemoryFactory(factory.Factory):
    class Meta:
        model = memories.Memory


class SwarmFactory(factory.Factory):
    class Meta:
        model = swarms.Swarm


class PopulationFactory(factory.Factory):
    class Meta:
        model = populations.Population


class PSOAgentFactory(factory.Factory):
    class Meta:
        model = agents.PSOAgent

    n_clusters = 1
    social_coefficient = 1
    personal_coefficient = 1
    inertia = 1


class GWOAgentFactory(factory.Factory):
    class Meta:
        model = agents.GWOAgent

    n_clusters = 1
