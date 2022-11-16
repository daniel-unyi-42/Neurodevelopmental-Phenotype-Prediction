import torch
from torch.nn import Module, Linear, BatchNorm1d, Sequential

class MLP(Module):
    def __init__(self, in_dim, hidden_dims, out_dim, device, act=torch.tanh):
        super(MLP, self).__init__()
        self.device = device
        self.lin1 = Linear(in_dim, hidden_dims[0])
        self.bn1 = BatchNorm1d(hidden_dims[0])
        self.lin2 = Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = BatchNorm1d(hidden_dims[1])
        self.lin3 = Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = BatchNorm1d(hidden_dims[2])
        self.lin4 = Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = BatchNorm1d(hidden_dims[3])
        self.lin5 = Linear(hidden_dims[3], hidden_dims[3])
        self.lin6 = Linear(hidden_dims[3], out_dim)
        self.act = act
        self.to(device)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(self.lin2(x))
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(self.lin3(x))
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(self.lin4(x))
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        # x_max, _ = torch.max(x, axis=1)
        x_mean = torch.mean(x, axis=1)
        # x_min, _ = torch.min(x, axis=1)
        x_c = torch.cat([x_mean], dim=1)
        x = self.act(self.lin5(x_c))
        return self.lin6(x)
