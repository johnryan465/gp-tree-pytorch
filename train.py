from typing import List, Tuple
from models.node import NodeModel
from models.tree import GPTree, GPTreeLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gpytorch
import torchvision
from tree.utils import BinaryTree
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from mnist import MNISTResNet


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
    model = node.data

    opt = gpytorch.optim.NGD(model.variational_parameters(), num_data=X.size(0), lr=0.1)
    
    elbo = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=X.size(0))

    opt.zero_grad()
    output = model(X)
    y = NodeModel.transform_target(node, y)
    loss = -elbo(output, y)
    loss.backward()
    opt.step()


def outer_step(tree: GPTree, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
    """
    A step for the outer layer of the optimzation
    """
    X, y = data

    opt = torch.optim.SGD([{"params": tree.hyperparameters()}], lr=0.1)
    elbo = GPTreeLoss(tree.likelihood, tree, num_data=X.size(0))

    opt.zero_grad()
    output = tree(X)
    loss = -elbo(output, y)
    loss.backward()

    opt.step()
    return loss.item()




def create_fe() -> torch.nn.Module:
    return MNISTResNet()


def train_fe(model, train, test):

    class Wrapper(nn.Module):
        def __init__(self, fe: torch.nn.Module) -> None:
            super().__init__()
            self.fe = fe
            self.fc2 = nn.Linear(1024, 10)

        def forward(self, x):
            x = self.fe(x)
            x = F.relu(x)
            return self.fc2(x)

    """
    Train the feature extractor to initialise the model
    """
    """
    A step for the outer layer of the optimzation
    """

    wrapped = Wrapper(model)
    opt = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    with tqdm(train, unit="batch") as pbar:
        for X, y in pbar:
            loss_fn = torch.nn.CrossEntropyLoss()
            opt.zero_grad()
            output = wrapped(X)
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
    batch_size = 64
    num_epochs = 1
    num_classes = 10
    per_class = 5
    train_loader, test_loader = create_dataset(batch_size)


    fe = create_fe()
    train_fe(fe, train_loader, test_loader)
    tree = create_tree(list(range(0, 10)))
    ind = get_inducing(fe, train_loader, num_classes*per_class)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("train_models"):
            model = GPTree(feature_extractor=fe, num_classes=num_classes, inducing_points=ind, tree=tree)
            for i in range(0, num_epochs):
                j = 0
                with tqdm(train_loader, unit="batch") as pbar:
                    for batch in pbar:
                        X, y = batch
                        embed = model.feature_extractor(X).detach()
                        for node in model.tree.node_list():
                            inner_step(node, (embed, y))
                        loss = outer_step(model, batch)
                        pbar.set_postfix(loss=loss)
                        j += 1
                        if j > 10:
                            break

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    

    # correct = 0
    # total = 0
    # for X, y in test_loader:
    #     l = model(X)
    #     s = next(iter(l))
    #    print(s.mean)
    #    res = model.likelihood.marginal(l).probs
    #    print(res)
    #    total += y.size(0)
    #    correct += torch.sum(torch.argmax(res, dim=1) == y)
    #    print(correct / total)

if __name__ == "__main__":
    with gpytorch.settings.num_likelihood_samples(512):
        main()