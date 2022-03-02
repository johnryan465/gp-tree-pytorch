from typing import List, Tuple
import gpytorch
from likelihoods.tree import GPTreeLikelihood
from gpytorch.distributions import Distribution
import torch
from torch.nn import ModuleList

from gpytorch.likelihoods import LikelihoodList
from gpytorch.lazy import BlockDiagLazyTensor, BlockInterleavedLazyTensor, CatLazyTensor, LazyTensor, lazify
from gptree.models.node import NodeModel
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood

from gptree.tree.utils import BinaryTree


class IterableMVN(MultitaskMultivariateNormal):
    def __init__(self, mvns: List[MultivariateNormal]):
        self.mvns = mvns

        if len(mvns) < 2:
            raise ValueError("Must provide at least 2 MVNs to form a MultitaskMultivariateNormal")
        if any(isinstance(mvn, MultitaskMultivariateNormal) for mvn in mvns):
            raise ValueError("Cannot accept MultitaskMultivariateNormals")
        if not all(m.batch_shape == mvns[0].batch_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same batch shape")
        if not all(m.event_shape == mvns[0].event_shape for m in mvns[1:]):
            raise ValueError("All MultivariateNormals must have the same event shape")
        mean = torch.stack([mvn.mean for mvn in mvns], -1)

        covar_blocks_lazy = CatLazyTensor(
            *[mvn.lazy_covariance_matrix.unsqueeze(0) for mvn in mvns], dim=0, output_device=mean.device
        )
        covar_lazy = BlockDiagLazyTensor(covar_blocks_lazy, block_dim=0)
        super().__init__(mean=mean, covariance_matrix=covar_lazy, interleaved=False)        

    def __iter__(self):
        return iter(self.mvns)


class GPTreeLoss(_ApproximateMarginalLogLikelihood):
    """
    Loss for the GP Tree
    """
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs).sum(-1)

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs).div(num_batch)
        kl_divergence = self.model.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior



class GPTree(gpytorch.models.GP):
    """
    We will want each of the sets of indexes for the nodes to be continous so we can 
    use narrow and reuse code
    """

    def __init__(self, feature_extractor: torch.nn.Module, num_classes: int, inducing_points, tree: BinaryTree):
        super().__init__()
        self.feature_dims = inducing_points.size(1)
        self.points_per_class = inducing_points.size(0) // num_classes
        inducing_points_ = torch.reshape(inducing_points.clone(), (num_classes, self.points_per_class, self.feature_dims))

        self.inducing_points =  torch.nn.Parameter(inducing_points_, requires_grad=True)
        self.feature_extractor = feature_extractor
        self.tree = self.initialise_tree(tree)
        models = list(map(lambda x: x.data, self.tree.node_list()))
        self.models = ModuleList(models)
        self.likelihood = GPTreeLikelihood(num_classes,  tree=tree)

    def initialise_tree(self, tree: BinaryTree[None]) -> BinaryTree[NodeModel]:
        """
        Attach GP to the nodes
        """
        left = self.initialise_tree(tree.left_node) if tree.left_node is not None else None
        right = self.initialise_tree(tree.right_node) if tree.right_node is not None else None
        start = min(list(tree.left_labels) + list(tree.right_labels))
        end = max(list(tree.left_labels) + list(tree.right_labels))
        model = NodeModel(torch.narrow(self.inducing_points, 0, start, (end-start)+1).view(-1, self.feature_dims))
        return BinaryTree(tree.left_labels, tree.right_labels, left, right, model)

    def kl_divergence(self):
        res = 0
        for model in self.models:
            res += model.variational_strategy.kl_divergence()

        return res

    def forward(self, x, **kwargs):
        embed = self.feature_extractor(x)
        results = list([model(embed) for model in self.models])
        return IterableMVN(results)
