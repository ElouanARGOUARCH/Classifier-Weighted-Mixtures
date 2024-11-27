from .conditional_CWM import *

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

class GaussianEncoder(torch.nn.Module):
    def __init__(self, d_latent, d_obs, hidden_dims):
        super().__init__()
        self.d_obs = d_obs
        self.d_latent = d_latent
        self.hidden_dims = hidden_dims

        network_dimensions = [self.d_obs] + hidden_dims + [2 * self.d_latent]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        self.encoder = torch.nn.Sequential(*network)

    def sample_latent(self, obs_values):
        out = self.encoder(obs_values)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        return torch.randn(obs_values.shape[0], self.d_latent).to(obs_values.device) * torch.exp(log_var / 2) + mu

    def log_prob(self, latent_values, obs_values):
        out = self.encoder(obs_values)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        assert latent_values.shape[0] == obs_values.shape[0], 'mismatch in number of samples'
        inside_exp = - torch.sum(torch.square(latent_values - mu) / torch.exp(log_var), dim=-1) / 2
        outside_exp = - self.d_obs * torch.log(torch.tensor(2 * torch.pi)) / 2 - torch.sum(log_var, dim=-1) / 2
        return outside_exp + inside_exp

    def standard_KL(self, obs_values):
        out = self.encoder(obs_values)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        return - torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=-1) / 2

    def encoder_loss(self, obs_values):
        return torch.sum(self.standard_KL(obs_values))

class GaussianDecoder(torch.nn.Module):
    def __init__(self, d_obs, d_latent, hidden_dimensions):
        super().__init__()
        self.d_latent = d_latent
        self.d_obs = d_obs
        self.hidden_dims = hidden_dimensions

        network_dimensions = [self.d_latent] + hidden_dimensions + [2 * self.d_obs]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        self.decoder = torch.nn.Sequential(*network)

    def sample_obs(self, latent_values):
        mu, log_var = torch.chunk(self.decoder(latent_values), 2, dim=-1)
        return torch.randn(latent_values.shape[0], self.d_obs).to(latent_values.device) * torch.exp(log_var / 2) + mu

    def log_prob(self, obs_values, latent_values):
        assert latent_values.shape[0] == obs_values.shape[0], 'mismatch in number of samples'
        mu, log_var = torch.chunk(self.decoder(latent_values), 2, dim=-1)
        return - torch.sum(torch.square(obs_values - mu) / torch.exp(log_var), dim=-1) / 2 - self.d_obs * torch.log(torch.tensor(2 * torch.pi)) / 2 - torch.sum(log_var, dim=-1) / 2

    def decoder_loss(self, obs_values, latent_values):
        assert latent_values.shape[0] == obs_values.shape[0], 'mismatch in number of samples'
        return -torch.sum(self.log_prob(obs_values, latent_values))

class BernouilliDecoder(torch.nn.Module):
    def __init__(self, d_obs, d_latent, hidden_dimensions):
        super().__init__()
        self.d_latent = d_latent
        self.d_obs = d_obs
        self.hidden_dims = hidden_dimensions

        network_dimensions = [self.d_latent] + hidden_dimensions + [self.d_obs]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.SiLU(), ])
        network.pop()
        network.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*network)

    def sample_obs(self, latent_values):
        return self.decoder(latent_values)

    def log_prob(self, obs_values, latent_values):
        assert latent_values.shape[:-1] == obs_values.shape[:-1], 'mismatch in number of samples'
        return -torch.sum(
            torch.nn.functional.binary_cross_entropy(self.decoder(latent_values), obs_values, reduction='none'), dim=-1)

    def decoder_loss(self, obs_values, latent_values):
        assert latent_values.shape[0] == obs_values.shape[0], 'mismatch in number of samples'
        return -self.log_prob(obs_values, latent_values).sum()

class CWMVAE(torch.nn.Module):
    def __init__(self, d_obs, d_latent, K=5, hidden_dimensions=[128, 128]):
        super().__init__()
        self.d_latent = d_latent
        self.d_obs = d_obs
        self.encoder = ConditionalClassifierWeightedMixture(d_latent,d_obs, K, hidden_dimensions)
        self.decoder = BernouilliDecoder(d_obs, d_latent, hidden_dimensions)
        self.prior = NormalReference(d_latent)

    def sample(self, num_samples):
        return self.decoder.sample_obs(self.prior.sample(num_samples))

    def log_prob(self, samples, mc_samples=100):
        return torch.logsumexp(self.decoder.log_prob(samples.unsqueeze(0).repeat(mc_samples, 1, 1),
                                                     self.prior.sample([mc_samples]).unsqueeze(1).repeat(1,
                                                                                                         samples.shape[
                                                                                                             0], 1)),
                               dim=0)

    def loss(self, samples):
        base_samples = self.encoder.reference.sample(samples.shape[:-1]).to(samples.device)
        inverse_T_obs = self.encoder.T.backward(base_samples, samples)
        log_weights = self.encoder.W.log_prob(torch.cat([base_samples, samples], dim=-1))
        return -torch.sum(torch.exp(log_weights) * (
                    self.decoder.log_prob(samples.unsqueeze(-2).repeat(1, self.encoder.K, 1),
                                          inverse_T_obs) + self.prior.log_prob(inverse_T_obs) - self.encoder.log_prob(
                inverse_T_obs, samples.unsqueeze(-2).repeat(1, self.encoder.K, 1))))

    def train(self, samples, epochs, batch_size=None, lr=1e-3, weight_decay=1e-9, verbose=False, trace_loss=False):
        para_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(para_list, lr=lr, weight_decay=weight_decay)

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
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str(
                    'loss = ' + str(round(iteration_loss / samples.shape[0], 10)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values

class VAE(torch.nn.Module):
    def __init__(self, d_obs,d_latent, hidden_dimensions = [128,128]):
        super().__init__()
        self.d_latent = d_latent
        self.d_obs = d_obs
        self.encoder = GaussianEncoder(d_latent, d_obs, hidden_dimensions)
        self.decoder = BernouilliDecoder(d_obs, d_latent, hidden_dimensions)
        self.prior = NormalReference(d_latent)


    def sample(self, num_samples):
        return self.decoder.sample_obs(self.prior.sample(num_samples))

    def log_prob(self, samples, mc_samples=100):
        return torch.logsumexp(self.decoder.log_prob(samples.unsqueeze(0).repeat(mc_samples, 1, 1),
                                                     self.prior.sample([mc_samples]).unsqueeze(1).repeat(1,
                                                                                                         samples.shape[
                                                                                                             0], 1)),
                               dim=0)

    def loss(self, samples):
        latent = self.encoder.sample_latent(samples)
        return self.encoder.encoder_loss(samples) + self.decoder.decoder_loss(samples, latent)

    def train(self, samples, epochs, batch_size=None, lr=1e-3, weight_decay=1e-9, verbose=False, trace_loss=False):
        para_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(para_list, lr=lr, weight_decay=weight_decay)

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
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0]) for i, batch in enumerate(dataloader)]).sum().item()
            if verbose:
                pbar.set_postfix_str(
                    'loss = ' + str(round(iteration_loss / samples.shape[0], 10)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values