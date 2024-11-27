import torch
from models import *
from datasets import *

train_samples, train_labels, test_samples, test_labels = get_MNIST_dataset(one_hot=True, visual=False, train_test_split = True)

#Display dataset samples
fig = plt.figure(figsize=(12,12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)
    ax.imshow(train_samples[train_labels[:, i % 10] == 1, :][i].reshape(28, 28))
plt.tight_layout()
plt.show()
fig.savefig("MNIST_Dataset.png", dpi = fig.dpi)

d_obs = train_samples.shape[-1]
d_latent = 200
K = 5
hidden_dims = [256,256]
epochs = 100
batch_size = 700
lr = 1e-3

model = VAE(d_obs, d_latent,hidden_dims)
model.train(train_samples, epochs, batch_size, lr, verbose = True)
print(model.log_prob(test_samples, mc_samples=100).mean())
torch.save(model, 'VAE_Gaussian_encoder_model.pt')
with torch.no_grad():
    VAE_samples = model.sample([16])
fig = plt.figure(figsize=(12,12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)
    ax.imshow(VAE_samples[i].reshape(28, 28))
plt.tight_layout()
plt.show()
fig.savefig("VAE_Gaussian_encoder_samples.png", dpi = fig.dpi)


model = CWMVAE(d_obs, d_latent,K, hidden_dims)
model.train(train_samples, epochs, batch_size, lr, verbose = True)
print(model.log_prob(test_samples, mc_samples=100).mean())
torch.save(model, 'VAE_CWM_encoder_model.pt')
with torch.no_grad():
    VAE_CWM_samples = model.sample([16])
fig = plt.figure(figsize=(12,12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)
    ax.imshow(VAE_samples[i].reshape(28, 28))
plt.tight_layout()
plt.show()
fig.savefig("VAE_CWM_encoder_samples.png", dpi = fig.dpi)