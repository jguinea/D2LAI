import numpy as np
import torch
from torch import nn
from torch.utils import data

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    print(X.shape)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    # Hay que meter el tensor en un wrapper de data para iterar sobre el
    dataset = data.TensorDataset(*data_arrays)
    # dataloader es el iterador
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    # hay que iterar sobre la respuesta de load array porque DataLoader
    # tiene yield. Por ejemplo sería: 
    # next(iter(load_array((features, labels), batch_size)))

# En Sequential metemos las capas de nuestra red neuronal para poder recorrerlas de una en una
# Linear es una capa fully connected. El primer parámetro es los inputs y el segundo lo outputs.
net = nn.Sequential(nn.Linear(2, 1))

# Inicializamos los parámetros del modelo:
# net[0] es la primera capa
# tiene weights y bias
# weights lo iniciamos sampleando una normal
net[0].weight.data.normal_(0, 0.01)
# data lo iniciamos como 0
net[0].bias.data.fill_(0)

# Definimos la loss function
loss = nn.MSELoss()

# Definimos el método de optimización (SDG)
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)


# El bucle que optimiza
batch_size = 10
num_epochs = 3
data_iter = load_array((features, labels), batch_size)

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
