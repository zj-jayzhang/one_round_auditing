import os
import pathlib
import random
import typing

import numpy as np
import scipy.stats
import torch

# Try to be as numerically stable as possible
LIRA_DTYPE_TORCH = torch.float64
LIRA_DTYPE_NUMPY = np.float64


def setup_seeds(
    seed: int,
    deterministic_algorithms: bool = True,
    benchmark_algorithms: bool = False,
):
    # Globally fix seeds in case manual seeding is missing somewhere
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic_algorithms:
        # Enable deterministic (GPU) operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        if benchmark_algorithms:
            raise ValueError("Benchmarking should not be enabled under deterministic algorithms")

    # NB: benchmarking significantly improves training speed in general,
    #  but can reduce performance if things like input shapes change a lot!
    torch.backends.cudnn.benchmark = benchmark_algorithms


def lira_attack(
    target_scores: torch.Tensor,
    shadow_scores: torch.Tensor,
    shadow_membership_mask: torch.Tensor,
    eps: float = 1e-30,
) -> torch.Tensor:
    assert shadow_scores.dim() == 3
    target_scores = target_scores.to(dtype=LIRA_DTYPE_TORCH)
    shadow_scores = shadow_scores.to(dtype=LIRA_DTYPE_TORCH)

    # Shadow scores are (num samples, num shadow models, num augmentations)
    shadow_in_indices = torch.stack([torch.argwhere(mask).squeeze(-1) for mask in shadow_membership_mask], dim=0)
    shadow_out_indices = torch.stack([torch.argwhere(mask).squeeze(-1) for mask in ~shadow_membership_mask], dim=0)
    scores_in = torch.gather(
        shadow_scores, dim=1, index=shadow_in_indices.unsqueeze(-1).tile((1, 1, shadow_scores.size(-1)))
    )
    scores_out = torch.gather(
        shadow_scores, dim=1, index=shadow_out_indices.unsqueeze(-1).tile((1, 1, shadow_scores.size(-1)))
    )
    means_in = torch.mean(scores_in, dim=1).numpy()
    means_out = torch.mean(scores_out, dim=1).numpy()
    stds_in = torch.std(scores_in, dim=1).numpy() + eps
    stds_out = torch.std(scores_out, dim=1).numpy() + eps
    target_scores = target_scores.numpy()
    log_prs_in = np.mean(scipy.stats.norm.logpdf(target_scores, means_in, stds_in), axis=-1)
    log_prs_out = np.mean(scipy.stats.norm.logpdf(target_scores, means_out, stds_out), axis=-1)
    result_scores = log_prs_in - log_prs_out
    assert result_scores.dtype == LIRA_DTYPE_NUMPY
    return torch.from_numpy(result_scores).to(dtype=LIRA_DTYPE_TORCH)

def lira_attack_audit(
    shadow_scores: torch.Tensor,
    shadow_membership_mask: torch.Tensor,
    global_variance: bool = False,
    eps: float = 1e-30,
    target_model_idx = 0
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    assert shadow_scores.dim() == 3
    num_samples, num_shadow_models, num_augmentations = shadow_scores.size()
    shadow_scores = shadow_scores.to(dtype=LIRA_DTYPE_TORCH)

    # full_attack_scores = torch.empty(num_samples * num_shadow_models, dtype=LIRA_DTYPE_TORCH)
    # full_attack_membership = torch.empty_like(full_attack_scores, dtype=torch.bool)
    current_target_scores = shadow_scores[:, target_model_idx, :]
    
    assert current_target_scores.size() == (num_samples, num_augmentations)
    current_shadow_scores = torch.cat(
        (shadow_scores[:, :target_model_idx], shadow_scores[:, target_model_idx + 1 :]),
        dim=1,
    )
    assert current_shadow_scores.size() == (num_samples, num_shadow_models - 1, num_augmentations)
    current_shadow_membership = torch.cat(
        (shadow_membership_mask[:, :target_model_idx], shadow_membership_mask[:, target_model_idx + 1 :]),
        dim=1,
    )
    # import pdb; pdb.set_trace()
    assert current_shadow_membership.size() == (num_samples, num_shadow_models - 1)
    # full_attack_membership[
    #     target_model_idx * num_samples : (target_model_idx + 1) * num_samples
    # ] = shadow_membership_mask[:, target_model_idx]

    # NB: Very annoyingly, # in and out scores now may differ per sample; do this non-vectorized...
    # Shadow scores are (num samples, num shadow models, num augmentations)
    means_in = torch.empty((num_samples, num_augmentations), dtype=LIRA_DTYPE_TORCH)
    means_out = torch.empty_like(means_in)

    if global_variance:
        current_full_mask = torch.tile(current_shadow_membership.unsqueeze(0), (num_augmentations, 1, 1))
        scores_in = (
            current_shadow_scores.permute(2, 0, 1)[current_full_mask].view(num_augmentations, -1).permute(1, 0)
        )
        scores_out = (
            current_shadow_scores.permute(2, 0, 1)[~current_full_mask].view(num_augmentations, -1).permute(1, 0)
        )
        # stds_in = (torch.std(scores_in, dim=0) + eps).unsqueeze(0)
        # stds_out = (torch.std(scores_out, dim=0) + eps).unsqueeze(0)
        # I think LiRA does this for global variance, i.e., also over all augmentations
        stds_in = torch.std(scores_in) + eps
        stds_out = torch.std(scores_out) + eps
    else:
        stds_in = torch.empty_like(means_in)
        stds_out = torch.empty_like(means_in)
    for sample_idx in range(num_samples):
        scores_in = current_shadow_scores[sample_idx, current_shadow_membership[sample_idx]]
        scores_out = current_shadow_scores[sample_idx, ~current_shadow_membership[sample_idx]]
        means_in[sample_idx] = torch.mean(scores_in, dim=0)
        means_out[sample_idx] = torch.mean(scores_out, dim=0)
        if not global_variance:
            stds_in[sample_idx] = torch.std(scores_in, dim=0) + eps
            stds_out[sample_idx] = torch.std(scores_out, dim=0) + eps

    means_in = means_in.numpy()
    means_out = means_out.numpy()
    stds_in = stds_in.numpy()
    stds_out = stds_out.numpy()
    target_scores = current_target_scores.numpy()
    log_prs_in = np.mean(scipy.stats.norm.logpdf(target_scores, means_in, stds_in), axis=-1)
    log_prs_out = np.mean(scipy.stats.norm.logpdf(target_scores, means_out, stds_out), axis=-1)
    result_scores = log_prs_in - log_prs_out
    assert result_scores.dtype == LIRA_DTYPE_NUMPY
    full_attack_scores = torch.from_numpy(
        result_scores
    ).to(dtype=LIRA_DTYPE_TORCH)
    full_attack_membership = shadow_membership_mask[:, target_model_idx]
    # 
    return full_attack_scores



def lira_attack_loo(
    shadow_scores: torch.Tensor,
    shadow_membership_mask: torch.Tensor,
    global_variance: bool = False,
    eps: float = 1e-30,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    assert shadow_scores.dim() == 3
    num_samples, num_shadow_models, num_augmentations = shadow_scores.size()
    shadow_scores = shadow_scores.to(dtype=LIRA_DTYPE_TORCH)

    full_attack_scores = torch.empty(num_samples * num_shadow_models, dtype=LIRA_DTYPE_TORCH)
    full_attack_membership = torch.empty_like(full_attack_scores, dtype=torch.bool)
    for target_model_idx in range(num_shadow_models):
        current_target_scores = shadow_scores[:, target_model_idx, :]
        assert current_target_scores.size() == (num_samples, num_augmentations)
        current_shadow_scores = torch.cat(
            (shadow_scores[:, :target_model_idx], shadow_scores[:, target_model_idx + 1 :]),
            dim=1,
        )
        assert current_shadow_scores.size() == (num_samples, num_shadow_models - 1, num_augmentations)
        current_shadow_membership = torch.cat(
            (shadow_membership_mask[:, :target_model_idx], shadow_membership_mask[:, target_model_idx + 1 :]),
            dim=1,
        )
        assert current_shadow_membership.size() == (num_samples, num_shadow_models - 1)
        full_attack_membership[
            target_model_idx * num_samples : (target_model_idx + 1) * num_samples
        ] = shadow_membership_mask[:, target_model_idx]

        # NB: Very annoyingly, # in and out scores now may differ per sample; do this non-vectorized...
        # Shadow scores are (num samples, num shadow models, num augmentations)
        means_in = torch.empty((num_samples, num_augmentations), dtype=LIRA_DTYPE_TORCH)
        means_out = torch.empty_like(means_in)

        if global_variance:
            current_full_mask = torch.tile(current_shadow_membership.unsqueeze(0), (num_augmentations, 1, 1))
            scores_in = (
                current_shadow_scores.permute(2, 0, 1)[current_full_mask].view(num_augmentations, -1).permute(1, 0)
            )
            scores_out = (
                current_shadow_scores.permute(2, 0, 1)[~current_full_mask].view(num_augmentations, -1).permute(1, 0)
            )
            # stds_in = (torch.std(scores_in, dim=0) + eps).unsqueeze(0)
            # stds_out = (torch.std(scores_out, dim=0) + eps).unsqueeze(0)
            # I think LiRA does this for global variance, i.e., also over all augmentations
            stds_in = torch.std(scores_in) + eps
            stds_out = torch.std(scores_out) + eps
        else:
            stds_in = torch.empty_like(means_in)
            stds_out = torch.empty_like(means_in)
        for sample_idx in range(num_samples):
            scores_in = current_shadow_scores[sample_idx, current_shadow_membership[sample_idx]]
            scores_out = current_shadow_scores[sample_idx, ~current_shadow_membership[sample_idx]]
            means_in[sample_idx] = torch.mean(scores_in, dim=0)
            means_out[sample_idx] = torch.mean(scores_out, dim=0)
            if not global_variance:
                stds_in[sample_idx] = torch.std(scores_in, dim=0) + eps
                stds_out[sample_idx] = torch.std(scores_out, dim=0) + eps

        means_in = means_in.numpy()
        means_out = means_out.numpy()
        stds_in = stds_in.numpy()
        stds_out = stds_out.numpy()
        target_scores = current_target_scores.numpy()
        log_prs_in = np.mean(scipy.stats.norm.logpdf(target_scores, means_in, stds_in), axis=-1)
        log_prs_out = np.mean(scipy.stats.norm.logpdf(target_scores, means_out, stds_out), axis=-1)
        result_scores = log_prs_in - log_prs_out
        assert result_scores.dtype == LIRA_DTYPE_NUMPY
        full_attack_scores[target_model_idx * num_samples : (target_model_idx + 1) * num_samples] = torch.from_numpy(
            result_scores
        ).to(dtype=LIRA_DTYPE_TORCH)
    return full_attack_scores, full_attack_membership


def hinge_score(raw_predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert raw_predictions.dim() >= 2 and labels.dim() == 1 and raw_predictions.size(0) == len(labels)
    raw_predictions = raw_predictions.to(dtype=LIRA_DTYPE_TORCH)

    target_predictions = raw_predictions[torch.arange(len(labels)), ..., labels]
    raw_predictions[torch.arange(len(labels)), ..., labels] = float("-inf")
    return target_predictions - torch.max(raw_predictions, dim=-1).values


def logit_score(raw_predictions: torch.Tensor, labels: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    assert raw_predictions.dim() >= 2 and labels.dim() == 1 and raw_predictions.size(0) == len(labels)
    raw_predictions = raw_predictions.to(dtype=LIRA_DTYPE_TORCH)

    # Original LiRA implementation first calculates probabilities via numerically stable softmax,
    # and then calculates the logit score from probabilities.
    # However, there is no need to calculate all probabilities, thereby avoiding log(exp(...)) operations
    # and using a potentially more appropriate LogSumExp normalization constant.

    # torch.logsumexp works with -inf, hence this version is more memory-efficient
    target_predictions = raw_predictions[torch.arange(len(labels)), ..., labels]
    raw_predictions[torch.arange(len(labels)), ..., labels] = float("-inf")
    return target_predictions - torch.logsumexp(raw_predictions, dim=-1)

    # FIXME: PyTorch does not use eps in log; do manual LSE implementation w/ epsilon if numerically unstable!
    # non_target_preds = raw_predictions[non_target_mask].view(
    #     raw_predictions.size()[:-1] + (raw_predictions.size(-1) - 1,)
    # )
    # max_non_target_preds = torch.max(non_target_preds, dim=-1).values
    # lse_non_target_preds = torch.log(
    #     torch.sum(
    #         torch.exp(non_target_preds - max_non_target_preds.unsqueeze(-1))
    #     )
    #     + eps
    # ) + max_non_target_preds
    # return raw_predictions[torch.arange(len(labels)), ..., labels] - lse_non_target_preds


def logit_score_from_probs(probs_predictions: torch.Tensor, labels: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    assert probs_predictions.dim() >= 2 and labels.dim() == 1 and probs_predictions.size(0) == len(labels)
    probs_predictions = probs_predictions.to(dtype=LIRA_DTYPE_TORCH).clone()

    # Directly get probabilities => no need to worry about normalizer and log(exp(...)) stuff,
    # can directly use the logit formula (as in original LiRA implementation)
    probs_target = probs_predictions[torch.arange(len(labels)), ..., labels]
    probs_predictions[torch.arange(len(labels)), ..., labels] = 0
    return torch.log(probs_target + eps) - torch.log(torch.sum(probs_predictions, dim=-1) + eps)



def get_top_canaries(canary_scores: torch.Tensor, shadow_membership_mask: torch.Tensor, canary_indices: np.ndarray, num_self_inf: int, output_dir: pathlib.Path):
    canary_membership_mask = shadow_membership_mask[canary_indices]  # num_canaries, num_shadow
    
    def calculate_influence_1(i, j, canary_scores, canary_membership_mask, num_self_inf):
        j_mask = canary_membership_mask[j]
        scores_in = canary_scores[i][j_mask]
        scores_out = canary_scores[i][~j_mask]
        if num_self_inf != 64:
            #! only use the num_self_inf models
            scores_in = scores_in[:num_self_inf]
            scores_out = scores_out[:num_self_inf]
        
        mu_A, sigma_A = np.mean(scores_in), np.std(scores_in)
        mu_B, sigma_B = np.mean(scores_out), np.std(scores_out)
        
        dist_in = scipy.stats.norm(mu_A, sigma_A)
        dist_out = scipy.stats.norm(mu_B, sigma_B)
        
        threshold = dist_out.ppf(1.0 - 0.001)
        inf_val = 1.0 - dist_in.cdf(threshold)

        return inf_val
    
    # use mean_in - mean_out as the influence score
    def calculate_influence_2(i, j, canary_scores, canary_membership_mask, num_self_inf):
        j_mask = canary_membership_mask[j]
        scores_in = canary_scores[i][j_mask]
        scores_out = canary_scores[i][~j_mask]
        if num_self_inf != 64:
            #! only use the num_self_inf models
            scores_in = scores_in[:num_self_inf]
            scores_out = scores_out[:num_self_inf]
        
        mu_A, mu_B = np.mean(scores_in), np.mean(scores_out)
        
        return mu_A - mu_B

    def get_influence_matrix(canary_scores, canary_membership_mask, num_self_inf):
        num_canaries = canary_scores.shape[0]
        influence = np.zeros((num_canaries, num_canaries))
        
        for i in range(num_canaries):
            for j in range(num_canaries):
                influence[i, j] = calculate_influence_1(i, j, canary_scores, canary_membership_mask, num_self_inf)
        
        return influence

    
    canary_scores = canary_scores[:, :64]
    canary_membership_mask = canary_membership_mask[:, :64]
    influence_matrix = get_influence_matrix(canary_scores.numpy(), canary_membership_mask.numpy(), num_self_inf)
    self_inf = np.diag(influence_matrix.copy())
    top_100 = np.argsort(self_inf)[-100:]
    print(np.sort(self_inf))
    return top_100