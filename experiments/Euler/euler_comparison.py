import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(output_dir, exist_ok=True)
import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
from models import ClassifierWeightedMixture, RealNVP, NeuralSplineFlow

# ---------------------------------------------------------------------------
# Helpers (reused from euler.py)
# ---------------------------------------------------------------------------

class logit():
    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def transform(self, x, alpha=None):
        assert torch.all(x <= 1) and torch.all(x >= 0)
        if alpha is None:
            alpha = self.alpha
        return torch.logit(alpha * torch.ones_like(x) + x * (1 - 2 * alpha))

    def inverse_transform(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return (torch.sigmoid(x) - alpha * torch.ones_like(x)) / (1 - 2 * alpha)

    def log_det(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return torch.sum(
            torch.log(
                (1 - 2 * alpha) * (
                    torch.reciprocal(alpha * torch.ones_like(x) + x * (1 - 2 * alpha))
                    + torch.reciprocal((1 - alpha) * torch.ones_like(x) - x * (1 - 2 * alpha))
                )
            ), dim=-1,
        )


def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class GreyScale2DImageDistribution():
    def __init__(self, file):
        self.rgb = image.imread(file)
        self.lines, self.columns = self.rgb.shape[:-1]
        self.grey = torch.tensor(rgb2gray(self.rgb))

    def sample(self, num_samples):
        vector_density = self.grey.flatten()
        vector_density = vector_density / torch.sum(vector_density)
        categorical_samples = torch.distributions.Categorical(probs=vector_density).sample(num_samples)
        return torch.cat([
            ((categorical_samples % self.columns + torch.rand(num_samples)) / self.columns).unsqueeze(-1),
            ((1 - (categorical_samples // self.columns + torch.rand(num_samples)) / self.lines)).unsqueeze(-1),
        ], dim=-1)


def plot_2d_function(f, range=[[-10, 10], [-10, 10]], bins=[50, 50], alpha=1, show=True):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0], range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        plt.pcolormesh(
            tt_x, tt_y,
            f(mesh).numpy().reshape(bins[0], bins[1]).T,
            cmap=matplotlib.colormaps['viridis'], alpha=alpha, lw=0,
        )
    if show:
        plt.show()


def plot_image_2d_points(samples, bins=(200, 200), range=None, alpha=1., show=True):
    assert samples.shape[-1] == 2
    hist, x_edges, y_edges = numpy.histogram2d(
        samples[:, 0].numpy(), samples[:, 1].numpy(), bins, range,
    )
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.pcolormesh(x_edges, y_edges, hist.T, cmap=matplotlib.colormaps['viridis'], alpha=alpha, lw=0)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

num_samples = 200000
training_epochs = 1000
batch_size = 20000
lr = 1e-2
K = 50
hidden_dims = [128, 128, 128]
pre_training_epochs = 100

print('CUDA available:', torch.cuda.is_available())

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

distribution = GreyScale2DImageDistribution(r"C:/Users/MB811VL/Projects/chapter/Classifier-Weighted-Mixtures/experiments/Euler/euler.jpg")
target_samples = distribution.sample([num_samples])
lines, columns = distribution.lines, distribution.columns

logit_transform = logit(alpha=1e-2)
transformed_samples = logit_transform.transform(target_samples)

# ---------------------------------------------------------------------------
# 1. CWM
# ---------------------------------------------------------------------------
print('\n=== Training CWM ===')
cwm = ClassifierWeightedMixture(2, K, hidden_dims=hidden_dims)
print(f'CWM parameters: {cwm.compute_number_params()}')
cwm.EM_pretraining(transformed_samples[:30000], pre_training_epochs, verbose=True)
cwm_loss = cwm.train(transformed_samples, training_epochs, batch_size, lr=lr, verbose=True, trace_loss=True)

# ---------------------------------------------------------------------------
# 2. Real NVP  (matched parameter budget via layer/hidden-dim tuning)
# ---------------------------------------------------------------------------
print('\n=== Training Real NVP ===')
realnvp = RealNVP(dim=2, n_layers=5, hidden_dims=[128, 128, 128])
print(f'Real NVP parameters: {realnvp.compute_number_params()}')
realnvp_loss = realnvp.train_model(
    transformed_samples, training_epochs, batch_size, lr=lr, verbose=True, trace_loss=True,
)

# ---------------------------------------------------------------------------
# 3. Neural Spline Flow
# ---------------------------------------------------------------------------
print('\n=== Training Neural Spline Flow ===')
nsf = NeuralSplineFlow(dim=2, n_layers=8, hidden_dims=[128,128,128,128], n_bins=8, tail_bound=4.0)
print(f'NSF parameters: {nsf.compute_number_params()}')
nsf_loss = nsf.train_model(
    transformed_samples, training_epochs, batch_size, lr=lr, verbose=True, trace_loss=True,
)

# ---------------------------------------------------------------------------
# Density plots (PDF evaluated on a grid, mapped back through the logit)
# ---------------------------------------------------------------------------

def make_pdf_fn(model):
    def pdf(x):
        return torch.exp(
            model.log_prob(logit_transform.transform(x)).squeeze(-1)
            + logit_transform.log_det(x)
        )
    return pdf


fig, axes = plt.subplots(1, 4, figsize=(28, 12))

plt.sca(axes[0])
plt.title('Target samples', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_image_2d_points(target_samples, bins=(lines, columns), show=True)

plt.sca(axes[1])
plt.title(f'CWM  ({cwm.compute_number_params()} params)', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_2d_function(make_pdf_fn(cwm), bins=(lines, columns), range=[[0., 1.], [0., 1.]], show=True)

plt.sca(axes[2])
plt.title(f'Real NVP  ({realnvp.compute_number_params()} params)', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_2d_function(make_pdf_fn(realnvp), bins=(lines, columns), range=[[0., 1.], [0., 1.]], show=True)

plt.sca(axes[3])
plt.title(f'NSF  ({nsf.compute_number_params()} params)', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_2d_function(make_pdf_fn(nsf), bins=(lines, columns), range=[[0., 1.], [0., 1.]], show=True)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'euler_comparison_density.png'), dpi=fig.dpi)
plt.show()

# ---------------------------------------------------------------------------
# Sample plots
# ---------------------------------------------------------------------------

with torch.no_grad():
    cwm_samples = logit_transform.inverse_transform(cwm.sample([num_samples]))
    realnvp_samples = logit_transform.inverse_transform(realnvp.sample(num_samples))
    nsf_samples = logit_transform.inverse_transform(nsf.sample(num_samples))

fig, axes = plt.subplots(1, 4, figsize=(28, 12))

plt.sca(axes[0])
plt.title('Target samples', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_image_2d_points(target_samples, bins=(lines, columns), show=False)

plt.sca(axes[1])
plt.title('CWM samples', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_image_2d_points(cwm_samples, bins=(lines, columns), show=False)

plt.sca(axes[2])
plt.title('Real NVP samples', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_image_2d_points(realnvp_samples, bins=(lines, columns), show=False)

plt.sca(axes[3])
plt.title('NSF samples', fontsize=16)
plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plot_image_2d_points(nsf_samples, bins=(lines, columns), show=False)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'euler_comparison_samples.png'), dpi=fig.dpi)
plt.show()

# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
epochs_range = range(1, training_epochs + 1)
ax.plot(epochs_range, [l / num_samples for l in cwm_loss], label='CWM')
ax.plot(epochs_range, [l / num_samples for l in realnvp_loss], label='Real NVP')
ax.plot(epochs_range, [l / num_samples for l in nsf_loss], label='NSF')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Negative log-likelihood (per sample)', fontsize=14)
ax.set_title('Training loss comparison', fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'euler_comparison_loss.png'), dpi=fig.dpi)
plt.show()

# ---------------------------------------------------------------------------
# Final test log-likelihood
# ---------------------------------------------------------------------------

test_samples = distribution.sample([50000])
transformed_test = logit_transform.transform(test_samples)

with torch.no_grad():
    cwm_nll = -cwm.log_prob(transformed_test).mean().item()
    realnvp_nll = -realnvp.log_prob(transformed_test).mean().item()
    nsf_nll = -nsf.log_prob(transformed_test).mean().item()

print('\n=== Test NLL (lower is better) ===')
print(f'CWM:      {cwm_nll:.4f}')
print(f'Real NVP: {realnvp_nll:.4f}')
print(f'NSF:      {nsf_nll:.4f}')

# Save models
torch.save(cwm.state_dict(), os.path.join(output_dir, 'cwm.pt'))
torch.save(realnvp.state_dict(), os.path.join(output_dir, 'realnvp.pt'))
torch.save(nsf.state_dict(), os.path.join(output_dir, 'nsf.pt'))
