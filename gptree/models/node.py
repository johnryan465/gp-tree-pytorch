import gpytorch
import torch
from gptree.tree.utils import BinaryTree
from gptree.likelihoods.node import PGLikelihood
from gptree.variational.tree_strategy import SharableLocationVariationalStrategy


class NodeModel(gpytorch.models.ApproximateGP):
    """
    GP model for each node in the Tree
    """

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = SharableLocationVariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy)
        self.likelihood = PGLikelihood()
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    @staticmethod
    def transform_target(model: BinaryTree["NodeModel"], targets: torch.Tensor) -> torch.Tensor:
        """
        As we have that the the indexes are continous we can transform the tensors actually prtty quick
        """
        # print(model_id)
        min_left = min(model.left_labels)
        max_left = max(model.left_labels)

        min_right = min(model.right_labels)
        max_right = max(model.right_labels)

        negative = (targets <= max_left) & (targets >= min_left)
        positive = (targets <= max_right) & (targets >= min_right)

        return positive.int() - negative.int()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
