import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Real NVP
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dims):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        d_in = mask.sum().int().item()
        d_out = dim - d_in
        self.s_net = MLP(d_in, d_out, hidden_dims)
        self.t_net = MLP(d_in, d_out, hidden_dims)

    def forward(self, x):
        x_masked = x[:, self.mask.bool()]
        s = self.s_net(x_masked)
        t = self.t_net(x_masked)
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        y_masked = y[:, self.mask.bool()]
        s = self.s_net(y_masked)
        t = self.t_net(y_masked)
        x = y.clone()
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        return x


class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dims=[128, 128]):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[:dim // 2] = 1.0
            if i % 2 == 1:
                mask = 1.0 - mask
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dims))

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z.pow(2).sum(-1) + self.dim * math.log(2 * math.pi))
        return log_pz + log_det

    def sample(self, num_samples):
        if isinstance(num_samples, list):
            num_samples = tuple(num_samples)
        z = torch.randn(num_samples, self.dim, device=next(self.parameters()).device) if isinstance(num_samples, int) else torch.randn((*num_samples, self.dim), device=next(self.parameters()).device)
        return self.inverse(z)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, x):
        return -torch.sum(self.log_prob(x))

    def train_model(self, samples, epochs, batch_size=None, lr=5e-3, weight_decay=5e-5, verbose=False, trace_loss=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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
                optimizer.zero_grad()
                batch_loss = self.loss(batch[0])
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0]) for i, batch in enumerate(dataloader)]
                    ).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss / samples.shape[0], 6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values


# ---------------------------------------------------------------------------
# Neural Spline Flow  (Rational-Quadratic splines, Durkan et al. 2019)
# ---------------------------------------------------------------------------

def rational_quadratic_spline_forward(x, widths, heights, derivatives, tail_bound=3.0):
    """Apply an RQ-spline transform element-wise.

    Args:
        x:           (...,) tensor of inputs
        widths:      (..., K) un-softmaxed bin widths
        heights:     (..., K) un-softmaxed bin heights
        derivatives: (..., K-1) un-softplussed interior knot derivatives
    Returns:
        y, log_det  same shape as x
    """
    K = widths.shape[-1]
    # Ensure valid parameters
    widths = F.softmax(widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(derivatives)
    # Pad derivatives at boundary (d_0 = d_K = 1)
    ones = torch.ones_like(derivatives[..., :1])
    derivatives = torch.cat([ones, derivatives, ones], dim=-1)  # (..., K+1)

    # Cumulative widths / heights -> knot positions
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)  # (..., K+1)
    cumwidths = cumwidths - tail_bound  # shift to [-B, B]

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = cumheights - tail_bound

    # Identity outside [-B, B]
    inside = (x >= -tail_bound) & (x <= tail_bound)
    x_clamped = torch.clamp(x, -tail_bound + 1e-6, tail_bound - 1e-6)

    # Find which bin each x falls into
    bin_idx = torch.searchsorted(cumwidths[..., 1:], x_clamped.unsqueeze(-1)).squeeze(-1)
    bin_idx = torch.clamp(bin_idx, 0, K - 1)

    # Gather bin parameters
    def _gather(t, idx):
        idx_exp = idx.unsqueeze(-1)
        return t.gather(-1, idx_exp).squeeze(-1)

    w_k = _gather(widths, bin_idx)
    h_k = _gather(heights, bin_idx)
    d_k = _gather(derivatives, bin_idx)
    d_k1 = _gather(derivatives[..., 1:], bin_idx)
    cw_k = _gather(cumwidths, bin_idx)
    ch_k = _gather(cumheights, bin_idx)

    xi = (x_clamped - cw_k) / w_k  # position within bin [0,1]
    xi = torch.clamp(xi, 1e-6, 1 - 1e-6)

    s_k = h_k / w_k
    numerator = h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
    denominator = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)
    y_inside = ch_k + numerator / denominator

    # Log derivative
    denom_sq = denominator.pow(2)
    log_det_inside = torch.log(s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2)) + 1e-8) - torch.log(denom_sq + 1e-8)

    y = torch.where(inside, y_inside, x)
    log_det = torch.where(inside, log_det_inside, torch.zeros_like(x))
    return y, log_det


def rational_quadratic_spline_inverse(y, widths, heights, derivatives, tail_bound=3.0):
    """Inverse of the RQ-spline transform."""
    K = widths.shape[-1]
    widths = F.softmax(widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(derivatives)
    ones = torch.ones_like(derivatives[..., :1])
    derivatives = torch.cat([ones, derivatives, ones], dim=-1)

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0) - tail_bound
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0) - tail_bound

    inside = (y >= -tail_bound) & (y <= tail_bound)
    y_clamped = torch.clamp(y, -tail_bound + 1e-6, tail_bound - 1e-6)

    bin_idx = torch.searchsorted(cumheights[..., 1:], y_clamped.unsqueeze(-1)).squeeze(-1)
    bin_idx = torch.clamp(bin_idx, 0, K - 1)

    def _gather(t, idx):
        return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    w_k = _gather(widths, bin_idx)
    h_k = _gather(heights, bin_idx)
    d_k = _gather(derivatives, bin_idx)
    d_k1 = _gather(derivatives[..., 1:], bin_idx)
    cw_k = _gather(cumwidths, bin_idx)
    ch_k = _gather(cumheights, bin_idx)

    s_k = h_k / w_k
    eta = (y_clamped - ch_k)

    a = h_k * (s_k - d_k) + eta * (d_k + d_k1 - 2 * s_k)
    b = h_k * d_k - eta * (d_k + d_k1 - 2 * s_k)
    c = -s_k * eta

    disc = b.pow(2) - 4 * a * c
    disc = torch.clamp(disc, min=0)
    xi = (2 * c) / (-b - torch.sqrt(disc + 1e-8))
    xi = torch.clamp(xi, 1e-6, 1 - 1e-6)

    x_inside = xi * w_k + cw_k
    x = torch.where(inside, x_inside, y)
    return x


class SplineCouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dims, n_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.register_buffer('mask', mask)
        d_in = mask.sum().int().item()
        d_out = dim - d_in
        # Each transformed dimension needs: K widths + K heights + (K-1) derivatives
        self.param_net = MLP(d_in, d_out * (3 * n_bins - 1), hidden_dims)

    def _get_params(self, x_masked, d_out):
        raw = self.param_net(x_masked)
        raw = raw.reshape(*x_masked.shape[:-1], d_out, 3 * self.n_bins - 1)
        widths = raw[..., :self.n_bins]
        heights = raw[..., self.n_bins:2 * self.n_bins]
        derivatives = raw[..., 2 * self.n_bins:]
        return widths, heights, derivatives

    def forward(self, x):
        mask_bool = self.mask.bool()
        x_masked = x[:, mask_bool]
        d_out = (~mask_bool).sum().int().item()
        widths, heights, derivatives = self._get_params(x_masked, d_out)
        x_transform = x[:, ~mask_bool]
        y_transform, log_det = rational_quadratic_spline_forward(
            x_transform, widths, heights, derivatives, self.tail_bound
        )
        y = x.clone()
        y[:, ~mask_bool] = y_transform
        return y, log_det.sum(dim=-1)

    def inverse(self, y):
        mask_bool = self.mask.bool()
        y_masked = y[:, mask_bool]
        d_out = (~mask_bool).sum().int().item()
        widths, heights, derivatives = self._get_params(y_masked, d_out)
        y_transform = y[:, ~mask_bool]
        x_transform = rational_quadratic_spline_inverse(
            y_transform, widths, heights, derivatives, self.tail_bound
        )
        x = y.clone()
        x[:, ~mask_bool] = x_transform
        return x


class NeuralSplineFlow(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dims=[128, 128], n_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[:dim // 2] = 1.0
            if i % 2 == 1:
                mask = 1.0 - mask
            self.layers.append(
                SplineCouplingLayer(dim, mask, hidden_dims, n_bins, tail_bound)
            )

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z.pow(2).sum(-1) + self.dim * math.log(2 * math.pi))
        return log_pz + log_det

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.dim, device=next(self.parameters()).device)
        return self.inverse(z)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, x):
        return -torch.sum(self.log_prob(x))

    def train_model(self, samples, epochs, batch_size=None, lr=5e-3, weight_decay=5e-5, verbose=False, trace_loss=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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
                optimizer.zero_grad()
                batch_loss = self.loss(batch[0])
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
            if verbose or trace_loss:
                with torch.no_grad():
                    iteration_loss = torch.tensor(
                        [self.loss(batch[0]) for i, batch in enumerate(dataloader)]
                    ).sum().item()
            if verbose:
                pbar.set_postfix_str('loss = ' + str(round(iteration_loss / samples.shape[0], 6)) + ' ; device: ' + str(device))
            if trace_loss:
                loss_values.append(iteration_loss)
        self.to(torch.device('cpu'))
        if trace_loss:
            return loss_values
