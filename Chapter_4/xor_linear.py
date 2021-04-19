import torch
from torch import nn
from torch.utils import data

features = [[0.,0.], [1.,0.], [0.,1.], [1., 1.]]
labels = [0, 1, 1, 0]

params = {'batch_size': 1,
          'shuffle': True}

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        sample = features, label
        return sample

features = torch.Tensor(features)
labels = torch.Tensor(labels)

training_set = ListDataset(features, labels)
training_generator = torch.utils.data.DataLoader(training_set, **params)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2,2))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

print(net)

batch_size, lr, num_epochs = 1, 0.01, 3
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)


for epoch in range(num_epochs):
    for X, y in training_generator:
        print(net(X))
        print(y)
        net_x = net(X)
        net_x = net_x.reshape(y.shape)
        l = loss(net(X).reshape(y.shape), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

