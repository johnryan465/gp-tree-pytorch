from typing import Tuple
from models.node import NodeModel
from models.tree import GPTree, GPTreeLoss
from sklearn import tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gpytorch
import torchvision
from tree.utils import BinaryTree


def create_dataset(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders
    """
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def inner_step(node: BinaryTree[NodeModel], data: Tuple[torch.Tensor, torch.Tensor]):
    """
    A step for the inner layer of the optimzation
    """
    X, y = data
    opt = node.data.optimizer
    
    elbo = gpytorch.mlls.VariationalELBO(node.data.likelihood, node.data, num_data=100, beta=0.5)

    opt.zero_grad()
    output = node.data(X)
    loss = -elbo(output, y)
    loss.backward(retain_graph=True)
    opt.step()


def outer_step(tree: GPTree, data: Tuple[torch.Tensor, torch.Tensor]):
    """
    A step for the outer layer of the optimzation
    """
    X, y = data
    tree.inducing_points.requires_grad = True
    opt = tree.optimizer
    elbo = GPTreeLoss(tree.likelihood, tree, num_data=100, beta=0.5)

    opt.zero_grad()
    output = tree(X)
    loss = -elbo(output, y)
    loss.backward()
    opt.step()
    tree.inducing_points.requires_grad = False




def create_fe() -> torch.nn.Module:
    class CNNMNIST(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
            self.fc1 = nn.Linear(1024, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1024)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return CNNMNIST()


def train_fe(model, train, test):
    """
    Train the feature extractor to initialise the model
    """
    pass


def create_tree() -> BinaryTree[None]:
    """
    Create the tree to pass to GP
    """
    return BinaryTree(0, [0], [1, 2], None, BinaryTree(1, [1], [2], None, None, None), None)


def get_inducing() -> torch.Tensor:
    """
    Return initial inducing points
    """
    return torch.ones(15, 10)


if __name__ == "__main__":
    batch_size = 64
    num_epochs = 1
    num_classes = 3
    train_loader, test_loader = create_dataset(64)
    ind = get_inducing()

    fe = create_fe()
    train_fe(fe, train_loader, test_loader)
    tree = create_tree()
    model = GPTree(feature_extractor=fe, num_classes=num_classes, inducing_points=ind, tree=tree)
    for i, batch in zip(range(0, num_epochs), iter(train_loader)):
        X, y = batch
        embed = model.feature_extractor(X)
        for node in model.tree.sorted_node_list():
            inner_step(node, (embed, y))
        outer_step(model, batch)
