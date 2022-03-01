from typing import List, Tuple
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
from tqdm import tqdm
from torchviz import make_dot


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
    opt = gpytorch.optim.NGD(node.data.variational_parameters(), num_data=X.size(0), lr=1)
    
    elbo = gpytorch.mlls.VariationalELBO(node.data.likelihood, node.data, num_data=X.size(0))

    opt.zero_grad()
    output = node.data(X)
    loss = -elbo(output, y)
    loss.backward()
    # print(node.left_labels, node.right_labels)
    # print(loss)
    opt.step()


def outer_step(tree: GPTree, data: Tuple[torch.Tensor, torch.Tensor]):
    """
    A step for the outer layer of the optimzation
    """
    X, y = data
    # tree.inducing_points.requires_grad = True
    # for param in tree.named_parameters():
    #     print(param)
    # print(tree.inducing_points)
    opt = torch.optim.SGD([{"params": tree.hyperparameters()}], lr=0.001)
    elbo = GPTreeLoss(tree.likelihood, tree, num_data=X.size(0))

    opt.zero_grad()
    output = tree(X)
    loss = -elbo(output, y)
    print(loss)
    dot = make_dot(loss, params=dict(tree.named_parameters()), show_attrs=True, show_saved=True)
    # dot.format = 'svg'
    dot.render()
    loss.backward()

    opt.step()
    # tree.inducing_points.requires_grad = False




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
    """
    A step for the outer layer of the optimzation
    """
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tqdm(train, unit="batch") as pbar:
        for X, y in pbar:
            loss_fn = torch.nn.CrossEntropyLoss()
            opt.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
            break




def create_tree(labels: List[int]) -> BinaryTree[None]:
    """
    Create the tree to pass to GP
    """
    mid = len(labels) // 2
    left_labels = labels[:mid]
    right_labels = labels[mid:]

    left_node = create_tree(left_labels) if len(left_labels) > 1 else None
    right_node = create_tree(right_labels) if len(right_labels) > 1 else None

    return BinaryTree(left_labels, right_labels, left_node, right_node, None)


def get_inducing(model, data, num: int) -> torch.Tensor:
    """
    Return initial inducing points
    """
    res = []
    count = 0
    for X, y in data:
        res.append(model(X))
        count += res[-1].size(0)
        if count > num:
            break

    res = torch.cat(res, dim=0)
    indices = torch.randperm(res.size(0))
    return res[indices[:num]].contiguous()



def main():
    batch_size = 256
    num_epochs = 1000
    num_classes = 10
    per_class = 5
    train_loader, test_loader = create_dataset(batch_size)


    fe = create_fe()
    train_fe(fe, train_loader, test_loader)
    tree = create_tree(list(range(0, 10)))
    ind = get_inducing(fe, train_loader, num_classes*per_class)
    model = GPTree(feature_extractor=fe, num_classes=num_classes, inducing_points=ind, tree=tree)
    for i, batch in zip(range(0, num_epochs), iter(train_loader)):
        X, y = batch
        embed = model.feature_extractor(X).detach()
        for node in model.tree.node_list():
            inner_step(node, (embed, y))
        outer_step(model, batch)

if __name__ == "__main__":
    main()