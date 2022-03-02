import torch
import gpytorch


class PGLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    = 1/2{2(y - 1/2)^T Knm Kmm^-1 - Tr(ΛQnn) - Tr(K^-1mm Kmn Λ Knm K^-1mm Σ) - µ^T K^-1 mm Kmn Λ Knm K^-1 mm µ},
    """
    # We expect labels {-1, 0, 1} where 0 means ignore
    # We will integrate the KL divergence between the Polya Gamma variables into the log prob
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        # function_dist
        # \mu = K_nm K_mm^-1 f^hat
        # \Sigma = K_nn - K_nm K^{-1}mm K_mn
        mean, variance = function_dist.mean, function_dist.variance
        raw_second_moment = variance + mean.pow(2)

        mask = observations == 0
        # print(observations)

        target = observations.to(mean.dtype)
        # print(variance)
        c = raw_second_moment.detach().sqrt()
        half_omega = 0.25 * torch.tanh(0.5 * c) / c

        # print(c)
        res = (0.5 * target * mean) - (half_omega * raw_second_moment)
        res = res * mask.to(res.dtype)
        # print(res)
        res = res.sum(dim=-1)

        return res

    # define the likelihood
    def forward(self, function_samples):
        return torch.distributions.Bernoulli(logits=function_samples)

    # define the marginal likelihood using Gauss Hermite quadrature
    def marginal(self, function_dist):
        prob_lambda = lambda function_samples: self.forward(function_samples).probs
        probs = self.quadrature(prob_lambda, function_dist)
        return torch.distributions.Bernoulli(probs=probs)
