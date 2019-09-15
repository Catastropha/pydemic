import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from pyriad.agents import PSOAgent
from pyriad.swarms import Swarm

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

agent = PSOAgent(n_clusters=3, social_coefficient=1.5, personal_coefficient=1.15, inertia=0.6)
swarm = Swarm()
swarm.populate(agent, 9)

tensor_data = torch.tensor(df.values).float()

swarm.train(tensor_data, 500)

labels, score = swarm.predict(tensor_data)

print(labels, score)
labels = labels.numpy().astype(str)
for i, _ in enumerate(labels):
    labels[i] += 'c'
df['predicted_species'] = labels
print(swarm.topK(1)[0].topK(1)[0])