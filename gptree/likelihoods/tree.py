from turtle import left
from typing import List, Optional
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.likelihood_list import LikelihoodList
from gpytorch.distributions import base_distributions
from likelihoods.node import PGLikelihood

from gptree.tree.utils import BinaryTree
import torch
import numpy as np


class GPTreeLikelihood(Likelihood):
    r"""
    Likelihood for GP Tree model
    """

    def __init__(self, num_classes: int, tree: BinaryTree):
        super().__init__()
        self.tree = tree
        self.num_classes = num_classes
        self.node_likelihood = PGLikelihood()

    def forward(self, function_samples, *params, **kwargs):
        likelihood_samples = [self.node_likelihood.forward(function_samples).probs for i in range(0, self.num_classes - 1)]
        nodes = self.tree.node_list()
        logit_contributions = torch.stack([likelihood_samples_to_logits(j, nodes[i], self.num_classes) for i, j in enumerate(likelihood_samples)], dim=0).sum(dim=0)
        res = base_distributions.Categorical(logits=logit_contributions)
        return res

    def transform_target(self, model_id: int, targets: torch.Tensor) -> torch.Tensor:
        """
        As we have that the the indexes are continous we can transform the tensors actually prtty quick
        """
        # print(model_id)
        model = self.tree.node_list()[model_id]
        min_left = min(model.left_labels)
        max_left = max(model.left_labels)

        min_right = min(model.right_labels)
        max_right = max(model.right_labels)

        negative = (targets <= max_left) & (targets >= min_left)
        positive = (targets <= max_right) & (targets >= min_right)

        return positive.int() - negative.int()

    def expected_log_prob(self, observations, function_dist, **kwargs):
        """
        We will convect the target to a target for each node, where if it doesn't matter it is 0
        """
        res = 0
        for i, dist in enumerate(function_dist):
            # print(i, dist)
            transformed_target = self.transform_target(i, observations)
            res += self.node_likelihood.expected_log_prob(transformed_target, dist)
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
