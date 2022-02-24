from typing import List, Optional
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.likelihood_list import LikelihoodList
from gpytorch.distributions import base_distributions

from gptree.tree.utils import BinaryTree, sorted_node_list
import torch
import numpy as np


class GPTreeLikelihood(Likelihood):
    r"""
    Likelihood for GP Tree model
    """

    def __init__(self, num_classes: int, tree: BinaryTree, likelihoods: LikelihoodList):
        super().__init__()
        self.tree = tree
        self.likelihoods = likelihoods
        self.num_classes = num_classes

    def forward(self, function_samples, *params, **kwargs):
        likelihood_samples = [self.likelihoods[i].forward(function_samples).probs for i in range(0, self.num_classes - 1)]
        nodes = sorted_node_list(self.tree)
        logit_contributions = torch.stack([likelihood_samples_to_logits(j, nodes[i], self.num_classes)
                                          for i, j in enumerate(likelihood_samples)], dim=0).sum(dim=0)
        # print(torch.nn.functional.softmax(logit_contributions)[:,:,0] - self.likelihoods[0].forward(function_samples).logits)
        # print(self.likelihoods[0].forward(function_samples).logits.shape)
        res = base_distributions.Categorical(logits=logit_contributions)
        return res

def node_to_log_np(prob: float, node: BinaryTree, num_classes: int) -> np.array:
    arr = np.zeros(num_classes)
    for idx in node.left_labels:
        arr[idx] = np.log(prob)
    for idx in node.right_labels:
        arr[idx] = np.log(1-prob)
    return arr


def likelihood_samples_to_logits(tensor: torch.Tensor, node: BinaryTree, num_classes: int) -> torch.Tensor:
    tensor = tensor.cpu().detach().numpy()
    tmp_fn = lambda x: node_to_log_np(x, node, num_classes)
    vf = np.vectorize(tmp_fn, signature='()->(n)')
    return torch.from_numpy(vf(tensor))
