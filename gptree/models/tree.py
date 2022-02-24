import gpytorch
from torch.nn import ModuleList

from gpytorch.likelihoods import LikelihoodList
from gptree.models.node import NodeModel

from gptree.tree.utils import BinaryTree, node_list, sorted_node_list
import torch


class GPTree(gpytorch.models.ApproximateGP):
    """
    We will want each of the sets of indexes for the nodes to be continous so we can 
    use more efficent slicing
    """

    def __init__(self, num_classes: int, inducing_points, tree: BinaryTree):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(0),
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.points_per_class = inducing_points.size(0) // num_classes
        models_l = [NodeModel(torch.ones(self.points_per_class*2, 1))]
        self.models = ModuleList(models_l)
        self.tree = tree
        self.likelihood = LikelihoodList(*[m.likelihood for m in self.models])

    def inner_variational_parameters(self):
        params = []
        for model in self.models:
            params += model.variational_parameters()
        return params
        
    def forward(self, x):
        results = []
        for node in sorted_node_list(self.tree):
            sm = min(node.left_labels)
            la = max(node.right_labels)
            res = self.models[node.id](x, inducing_points=self.variational_strategy.inducing_points
                                 [sm * self.points_per_class: (la+1) * self.points_per_class])
            results.append(res)

        return results[0] # gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(results)
