import torch
from torch import nn

class LocationScale(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.K = K
        self.dim = dim

        self.m = nn.Parameter(torch.randn(self.K, self.dim))
        self.log_s = nn.Parameter(torch.zeros(self.K, self.dim))

    def backward(self, z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X-self.m.expand_as(X))/torch.exp(self.log_s).expand_as(X)

    def log_det_J(self,x):
        return -self.log_s.sum(-1)

class ConditionalLocationScale(torch.nn.Module):
    def __init__(self,dim,cond_dim, K, hidden_dimensions):
        super().__init__()
        self.K = K
        self.dim = dim #dimension of distribution, i.e. of the variable x
        self.cond_dim = cond_dim  #dimension of conditioning variable y

        self.network_dimensions = [self.cond_dim] + hidden_dimensions + [2*self.K*self.dim]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1),torch.nn.Tanh(),])
        network.pop()
        self.f = torch.nn.Sequential(*network)

    def backward(self, z, y, return_log_det = False):
        assert z.shape[:-1]==y.shape[:-1], 'number of z samples does not match the number of theta samples'
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.dim
        m, log_s = torch.chunk(torch.reshape(self.f(y), new_desired_size), 2, -1)
        if return_log_det:
            return Z * torch.exp(log_s).expand_as(Z) + m.expand_as(Z), -log_s.sum(-1)
        else:
            return Z * torch.exp(log_s).expand_as(Z) + m.expand_as(Z)

    def forward(self, x, y, return_log_det = False):
        assert x.shape[:-1]==y.shape[:-1], 'number of x samples does not match the number of theta samples'
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.dim
        m, log_s = torch.chunk(torch.reshape(self.f(y), new_desired_size),2,-1)
        if return_log_det:
            return (X-m.expand_as(X))/torch.exp(log_s).expand_as(X), -log_s.sum(-1)
        else:
            return (X - m.expand_as(X)) / torch.exp(log_s).expand_as(X)

    def log_det_J(self,x, y):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        new_desired_size = desired_size
        new_desired_size[-1] = 2*self.dim
        m, log_s = torch.chunk(torch.reshape(self.f(y), new_desired_size),2,-1)
        return -log_s.sum(-1)

