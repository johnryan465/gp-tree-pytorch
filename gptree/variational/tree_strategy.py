from gpytorch.variational import VariationalStrategy


class SharableLocationVariationalStrategy(VariationalStrategy):
    """
    A variational strategy to allow for the sharing of inducing points
    Basically we tell not to learn the inducing points,
    Delete the created buffer and replace it with the original tensor.
    """
    def __init__(self, model, inducing_points, variational_distribution):
        super().__init__(model, inducing_points, variational_distribution, False)
        self._buffers["inducing_points"] = inducing_points
