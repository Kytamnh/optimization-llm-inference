from __future__ import annotations

import torch


def build_ref_point(train_y: torch.Tensor):
    mins = train_y.min(dim=0).values
    maxs = train_y.max(dim=0).values
    span = (maxs - mins).abs().clamp_min(1e-3)
    return (mins - 0.1 * span).tolist()


def build_qnehvi(model, train_x: torch.Tensor, train_y: torch.Tensor):
    if model is None or train_x is None or train_x.shape[0] < 2:
        return None
    try:
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement,
        )
        from botorch.sampling.normal import SobolQMCNormalSampler
    except Exception:
        return None

    try:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    except TypeError:  # pragma: no cover - older botorch compatibility
        sampler = SobolQMCNormalSampler(num_samples=128)

    return qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=build_ref_point(train_y),
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
    )
