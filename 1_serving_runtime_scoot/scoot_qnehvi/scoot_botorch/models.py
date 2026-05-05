from __future__ import annotations

import warnings

import torch


def _import_botorch():
    try:
        from botorch.fit import fit_gpytorch_mll
    except Exception:  # pragma: no cover - compatibility with older botorch
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll
    try:
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    except Exception:  # pragma: no cover
        MixedSingleTaskGP = None
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    return fit_gpytorch_mll, MixedSingleTaskGP, ModelListGP, SingleTaskGP, Standardize, ExactMarginalLogLikelihood


def fit_objective_model(train_x: torch.Tensor, train_y: torch.Tensor, categorical_dims):
    if train_x is None or train_y is None or train_x.shape[0] < 2:
        return None

    (
        fit_gpytorch_mll,
        MixedSingleTaskGP,
        ModelListGP,
        SingleTaskGP,
        Standardize,
        ExactMarginalLogLikelihood,
    ) = _import_botorch()

    models = []
    for i in range(train_y.shape[-1]):
        y = train_y[:, i : i + 1]
        if MixedSingleTaskGP is not None:
            model = MixedSingleTaskGP(
                train_x,
                y,
                cat_dims=list(categorical_dims),
                outcome_transform=Standardize(m=1),
            )
        else:
            model = SingleTaskGP(train_x, y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        models.append(model)
    return ModelListGP(*models)
