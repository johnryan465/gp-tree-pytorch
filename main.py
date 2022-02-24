import torch


from torch.utils.data import TensorDataset, DataLoader


import gpytorch
import os
import math
from math import floor
from gptree.likelihoods.node import PGLikelihood
from gptree.likelihoods.tree import GPTreeLikelihood

from gptree.models.node import NodeModel
from gptree.models.tree import GPTree
from gptree.tree.utils import BinaryTree
from tqdm import tqdm
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal


if __name__ == "__main__":

    # this is for running the notebook in our testing framework
    smoke_test = ('CI' in os.environ)

    N = 100
    X = torch.linspace(-1., 1., N)
    probs = (torch.sin(X * math.pi).add(1.).div(2.))
    y = torch.distributions.Bernoulli(probs=probs).sample()
    X = X.unsqueeze(-1)

    train_n = int(floor(0.8 * N))
    indices = torch.randperm(N)
    train_x = X[indices[:train_n]].contiguous()
    train_y = y[indices[:train_n]].contiguous()

    test_x = X[indices[train_n:]].contiguous()
    test_y = y[indices[train_n:]].contiguous()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=100000, shuffle=False)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    tree = BinaryTree(id=0, left_labels=[0], right_labels=[1], left_node=None, right_node=None)
    model = GPTree(inducing_points=train_x[:500, :], tree=tree, num_classes=2)
    likelihood = GPTreeLikelihood(2, tree=tree, likelihoods=[PGLikelihood()])

    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=1)
    variational_ngd_optimizer_2 = gpytorch.optim.NGD(model.inner_variational_parameters(), num_data=train_y.size(0), lr=0.1)

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    num_epochs = 1 if smoke_test else 100
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)

        for x_batch, y_batch in minibatch_iter:
            # Perform NGD step to optimize variational parameters
            variational_ngd_optimizer.zero_grad()
            # variational_ngd_optimizer_2.zero_grad()
            hyperparameter_optimizer.zero_grad()

            output = model(x_batch)
            print(output.loc)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            variational_ngd_optimizer.step()
            # variational_ngd_optimizer_2.step()
            hyperparameter_optimizer.step()

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            print(preds.mean)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]
    print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))
