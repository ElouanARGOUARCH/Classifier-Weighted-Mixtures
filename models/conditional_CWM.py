from .CWM import *

class ConditionalClassifierWeightedMixture(torch.nn.Module):
    def __init__(self, dim,cond_dim, K, hidden_dimensions):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.K = K

        self.reference = NormalReference(dim)

        self.W = SoftmaxWeight(dim+cond_dim,K, hidden_dimensions)

        self.T = ConditionalLocationScale(dim,cond_dim,K, hidden_dimensions)

    def compute_log_v(self,x, y):
        assert x.shape[:-1] == y.shape[:-1], 'wrong shapes'
        theta_unsqueezed = y.unsqueeze(-2).repeat(1, self.K, 1)
        z = self.T.forward(x, y)
        log_v = self.reference.log_prob(z) + torch.diagonal(self.W.log_prob(torch.cat([z, theta_unsqueezed], dim = -1)), 0, -2, -1) + self.T.log_det_J(x, y)
        return log_v - torch.logsumexp(log_v, dim = -1, keepdim= True)

    def sample_latent(self,x, y):
        assert x.shape[:-1] == y.shape[:-1], 'wrong shapes'
        z = self.T.forward(x, y)
        pick = torch.distributions.Categorical(torch.exp(self.compute_log_v(x, y))).sample()
        return z[range(z.shape[0]), pick, :]

    def log_prob(self, x, y):
        assert x.shape[:-1] == y.shape[:-1], 'wrong shapes'
        desired_size = list(y.shape)
        desired_size.insert(-1, self.K)
        z = self.T.forward(x, y)
        return torch.logsumexp(self.reference.log_prob(z) + torch.diagonal(self.W.log_prob(torch.cat([z, y.unsqueeze(-2).expand(desired_size)], dim = -1)), 0, -2, -1)+ self.T.log_det_J(x, y),dim=-1)

    def sample(self, y):
        z = self.reference.sample(y.shape[0])
        x = self.T.backward(z, y)
        pick = torch.distribution.Categorical(torch.exp(self.W.log_prob(torch.cat([z, y], dim = -1)))).sample()
        return x[range(x.shape[0]), pick, :]

    def initialize_with_EM(self, x, epochs, verbose=False):
        em = DiagGaussianMixtEM(self.dim, self.K)
        em.train(x,epochs, verbose)
        self.T.f[-1].weight = torch.nn.Parameter(
            torch.zeros(self.T.network_dimensions[-1], self.T.network_dimensions[-2]))
        self.T.f[-1].bias = torch.nn.Parameter(torch.cat([em.m, em.log_s], dim=-1).flatten())
        self.W.f[-1].weight = torch.nn.Parameter(
            torch.zeros(self.W.network_dimensions[-1], self.W.network_dimensions[-2]))
        self.W.f[-1].bias = torch.nn.Parameter(em.log_pi)

    def loss(self, x, y):
        z, log_det_J = self.T.forward(x, y, return_log_det=True)
        return -torch.mean(torch.logsumexp(self.reference.log_prob(z) + torch.diagonal(self.W.log_prob(torch.cat([z, y.unsqueeze(-2).repeat(1, self.K, 1)], dim = -1)), 0, -2, -1) + log_det_J, dim=-1))

    def train(self,x,y,epochs, batch_size = None, lr = 5e-3, weight_decay = 5e-5, verbose = False, trace_loss = False):
        assert x.shape[:-1] == y.shape[:-1], 'mismatch number of samples'

        self.para_list = list(self.parameters())
        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-3)

        if batch_size is None:
            batch_size = self.x_samples.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
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
                batch_loss = self.loss(batch[0],batch[1])
                batch_loss.backward()
                self.optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor([self.loss(batch[0],batch[1]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss/x.shape[0],6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values