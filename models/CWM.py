import torch
from tqdm import tqdm
from .reference import *
from .weights import *
from .invertible_mappings import *

class DiagGaussianMixtEM(torch.nn.Module):
    def __init__(self,dim,K):
        super().__init__()

        self.dim = dim
        self.K = K
        self.log_pi = torch.log(torch.ones([self.K])/self.K)
        self.m = torch.randn(K, dim)
        self.log_s = torch.randn(K, dim)
        self.reference= NormalReference(dim)

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X - self.m.expand_as(X)) / torch.exp(self.log_s).expand_as(X)

    def backward(self,z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def log_det_J(self,x):
        return -torch.sum(self.log_s, dim = -1)

    def compute_log_v(self,x):
        z = self.forward(x)
        unormalized_log_v = self.reference.log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1)+ self.log_det_J(x)
        return unormalized_log_v - torch.logsumexp(unormalized_log_v, dim = -1, keepdim= True)

    def sample_latent(self,x, joint = False):
        z = self.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        if not joint:
            return z[range(z.shape[0]), pick, :]
        else:
            return z[range(z.shape[0]), pick, :],pick

    def log_prob(self, x):
        z = self.forward(x)
        return torch.logsumexp(self.reference.log_prob(z) + self.log_pi.unsqueeze(0).repeat(x.shape[0],1) + self.log_det_J(x),dim=-1)

    def sample(self, num_samples, joint=False):
        z = self.reference.sample(num_samples)
        x = self.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.log_pi.unsqueeze(0).repeat(x.shape[0],1))).sample()
        if not joint:
            return x[range(x.shape[0]), pick, :]
        else:
            return x[range(x.shape[0]), pick, :],pick

    def M_step(self, x):
        v = torch.exp(self.compute_log_v(x))
        c = torch.sum(v, dim=0)
        self.log_pi = torch.log(c) - torch.logsumexp(torch.log(c), dim = 0)
        self.m = torch.sum(v.unsqueeze(-1).repeat(1, 1, self.dim) * x.unsqueeze(-2).repeat(1, self.K, 1),
                                dim=0) / c.unsqueeze(-1)
        temp2 = torch.square(x.unsqueeze(1).repeat(1,self.K, 1) - self.m.unsqueeze(0).repeat(x.shape[0],1,1))
        self.log_s = torch.log(torch.sum(v.unsqueeze(-1).repeat(1, 1, self.dim) * temp2,dim=0)/c.unsqueeze(-1))/2

    def train(self,samples, epochs, verbose = False, trace_loss = False):
        self.log_pi = torch.log(torch.ones([self.K]) / self.K)
        self.m = samples[torch.randint(low=0, high=samples.shape[0], size=[self.K])]
        self.log_s = torch.log(torch.var(samples, dim=0)).unsqueeze(0).repeat(self.K, 1) / 2
        if trace_loss:
            loss_values = []
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            self.M_step(samples)
            if verbose or trace_loss:
                loss = -torch.sum(self.log_prob(samples)).item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(loss/samples.shape[0]))
            if trace_loss:
                loss_values.append(loss)
        if trace_loss:
            return loss_values

class ClassifierWeightedMixture(torch.nn.Module):
    def __init__(self, dim, K,hidden_dims = []):
        super().__init__()
        self.dim = dim
        self.K = K
        self.reference_distribution = NormalReference(self.dim)

        self.W = SoftmaxWeight(self.dim, self.K, hidden_dims)
        self.T = LocationScale(self.dim, self.K)

    def EM_pretraining(self,samples,epochs, verbose = False):
        em = DiagGaussianMixtEM(self.dim,self.K)
        em.train(samples,epochs, verbose)
        self.T.m = torch.nn.Parameter(em.m)
        self.T.log_s = torch.nn.Parameter(em.log_s)
        self.W.f[-1].weight = torch.nn.Parameter(torch.zeros(self.K,self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)
        self.reference_distribution = NormalReference(self.dim)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_log_v(self,x):
        z = self.T.forward(x)
        log_v = self.reference_log_prob(z) + torch.diagonal(self.W.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x):
        z = self.T.forward(x)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_prob(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(self.reference_distribution.log_prob(z) + torch.diagonal(self.W.log_prob(z),0,-2,-1) + self.T.log_det_J(x),dim=-1)

    def sample(self, num_samples):
        z = self.reference_distribution.sample(num_samples)
        x = self.T.backward(z)
        pick = torch.distributions.Categorical(torch.exp(self.W.log_prob(z))).sample()
        return x[range(x.shape[0]), pick, :]

    def loss(self, x):
        return -torch.sum(self.log_prob(x))

    def train(self, samples, epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False, trace_loss = False):
        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=lr, weight_decay=weight_decay)

        if batch_size is None:
            batch_size = samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(samples.to(device))
        if trace_loss:
            loss_values = []
        if verbose:
            pbar = tqdm(range(epochs))
        else:
            pbar = range(epochs)
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                batch_loss = self.loss(batch[0])
                batch_loss.backward()
                self.optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss/samples.shape[0],6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values