import gpytorch
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
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.optimizer = gpytorch.optim.NGD(self.variational_parameters(), num_data=64, lr=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
