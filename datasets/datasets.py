import torch
import torchvision
import matplotlib.pyplot as plt

def get_MNIST_dataset(one_hot = False,repository = '../../datasets/data', visual = False, train_test_split = False):
    mnist_trainset = torchvision.datasets.MNIST(root=repository, train=True,
                                    download=True, transform=None)
    mnist_testset = torchvision.datasets.MNIST(root=repository, train=False,
                                   download=True, transform=None)

    train_labels = mnist_trainset.targets
    test_labels = mnist_testset.targets
    if one_hot:
        train_labels = torch.nn.functional.one_hot(train_labels)
        test_labels = torch.nn.functional.one_hot(test_labels)

    temp_train = mnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = mnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256

    if not train_test_split:
        samples = torch.cat([torch.cat([train_samples, test_samples], dim = 0)])
        labels = torch.cat([torch.cat([train_labels, test_labels], dim = 0)])
    if visual:
        for i in range(16):
            ax = plt.subplot(4,4,i+1)
            ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            if one_hot:
                ax.imshow(train_samples[train_labels[:,i % 10,] == 1, :][i].reshape(28, 28))
            else:
                ax.imshow(train_samples[train_labels==i%10,:][i].reshape(28,28))
        plt.tight_layout()
        plt.show()
    if not train_test_split:
        return samples, labels
    else:
        return train_samples,train_labels,test_samples, test_labels

def get_FashionMNIST_dataset(one_hot = False,repository = '../../datasets/data', visual = False):
    fmnist_trainset = torchvision.datasets.FashionMNIST(root=repository, train=True,
                                            download=True, transform=None)
    fmnist_testset = torchvision.datasets.FashionMNIST(root=repository, train=False,
                                           download=True, transform=None)
    train_labels = fmnist_trainset.targets
    test_labels = fmnist_testset.targets
    temp_train = fmnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = fmnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
    if visual:
        for i in range(16):
            ax = plt.subplot(4,4,i+1)
            ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            if one_hot:
                ax.imshow(train_samples[train_labels[:,i % 10,] == 1, :][i].reshape(28, 28))
            else:
                ax.imshow(train_samples[train_labels == i % 10, :][i].reshape(28, 28))
        plt.show()
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)