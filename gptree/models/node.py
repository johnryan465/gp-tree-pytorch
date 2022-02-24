import gpytorch
from gptree.likelihoods.node import PGLikelihood

from gptree.variational.inducing_point_shared import SharedInducingPointsVariational



class NodeModel(gpytorch.models.ApproximateGP):
    """
    GP model for each node in the Tree
    """
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = SharedInducingPointsVariational(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super(NodeModel, self).__init__(variational_strategy)
        self.likelihood = PGLikelihood()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)