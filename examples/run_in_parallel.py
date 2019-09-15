import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from pyriad.agents import PSOAgent, GWOAgent
from pyriad.swarms import Swarm
from pyriad.populations import Population

df = pd.read_csv('data/iris.csv')
df = df.drop(columns=['Id', 'Species'])

scaler = StandardScaler()

values = df['SepalLengthCm'].values
values = values.reshape((len(values), 1))
scaler = scaler.fit(values)
df['SepalLengthCm'] = scaler.transform(values)

values = df['SepalWidthCm'].values
values = values.reshape((len(values), 1))
scaler = scaler.fit(values)
df['SepalWidthCm'] = scaler.transform(values)

values = df['PetalLengthCm'].values
values = values.reshape((len(values), 1))
scaler = scaler.fit(values)
df['PetalLengthCm'] = scaler.transform(values)

values = df['PetalWidthCm'].values
values = values.reshape((len(values), 1))
scaler = scaler.fit(values)
df['PetalWidthCm'] = scaler.transform(values)

tensor_data = torch.tensor(df.values).float()

agent = PSOAgent(n_clusters=3, social_coefficient=1.5, personal_coefficient=1.15, inertia=0.6)
pso_swarm = Swarm()
pso_swarm.populate(agent, 9)

agent = GWOAgent(n_clusters=3)
gwo_swarm = Swarm(memory_size=3)
gwo_swarm.populate(agent, 19)

population = Population()
population.add(pso_swarm)
population.add(gwo_swarm)
population.run(data=tensor_data, episodes=500)
