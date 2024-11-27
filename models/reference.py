import torch

class NormalReference(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def sample(self, num_samples):
        temp = list(num_samples)
        temp.append(self.p)
        return torch.randn(temp)

    def log_prob(self, z):
        return -torch.sum(torch.square(z), dim = -1)/2 - self.p*torch.log(torch.tensor(2 * torch.pi))/2