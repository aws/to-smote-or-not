# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
)

from smote_variants import polynom_fit_SMOTE


OVERSAMPLER_HPS = {
    "none": [],
    "default": [],
    "random": [],
    "smote": [
        {"k_neighbors": 3},
        {"k_neighbors": 5},
        {"k_neighbors": 7},
        {"k_neighbors": 9},
    ],
    "svm": [
        {"k_neighbors": 4, "m_neighbors": 8},
        {"k_neighbors": 4, "m_neighbors": 12},
        {"k_neighbors": 6, "m_neighbors": 8},
        {"k_neighbors": 6, "m_neighbors": 12},
    ],
    "adasyn": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 7},
        {"n_neighbors": 9},
    ],
    "border": [
        {"k_neighbors": 4, "m_neighbors": 8},
        {"k_neighbors": 4, "m_neighbors": 12},
        {"k_neighbors": 6, "m_neighbors": 8},
        {"k_neighbors": 6, "m_neighbors": 12},
    ],
    "poly": [{"topology": "star"}, {"topology": "bus"}, {"topology": "mesh"}],
}


def _poly_oversample(
    x: np.ndarray, y: np.ndarray, params: dict, seed: int, ratio: float
):
    sampling_strategy = _imblearn_sampling_strategy(y, ratio)
    if not sampling_strategy:
        return None, None
    try:
        n_maj = np.sum(y == 0)
        n_min = np.sum(y == 1)
        desired_n_min = sampling_strategy[1]
        proportion = (desired_n_min - n_min) / (n_maj - n_min)
        if proportion <= 0:
            return None, None
        return polynom_fit_SMOTE(
            random_state=seed, proportion=proportion, **params
        ).sample(x, y.ravel())
    except (ValueError, RuntimeError):
        return None, None


def _imblearn_oversample(
    x: np.ndarray,
    y: np.ndarray,
    oversampler_name: str,
    params: dict,
    seed: int,
    ratio: float,
):
    sampling_strategy = _imblearn_sampling_strategy(y, ratio)
    if not sampling_strategy:
        return None, None
    try:
        oversampler = {
            "smote": SMOTE,
            "border": BorderlineSMOTE,
            "svm": SVMSMOTE,
            "adasyn": ADASYN,
        }[oversampler_name]
        return oversampler(
            random_state=seed, sampling_strategy=sampling_strategy, **params
        ).fit_resample(x, y)
    except (ValueError, RuntimeError) as e:
        print(e)
        return None, None


def _random_oversample(x: np.ndarray, y: np.ndarray, seed: int, ratio: float):
    sampling_strategy = _imblearn_sampling_strategy(y, ratio)
    if not sampling_strategy:
        return None, None
    return RandomOverSampler(
        random_state=seed, sampling_strategy=sampling_strategy
    ).fit_resample(x, y)


def _imblearn_sampling_strategy(y: np.ndarray, ratio: float):
    # Calcualte sampling strategy. 0 Must be the majority label and 1 must be the minority label.
    count0 = np.sum(y == 0)
    sampling_strategy = {0: count0, 1: int(np.round(count0 * ratio / (1 - ratio)))}
    if sampling_strategy[1] < np.sum(y == 1):
        return None
    return sampling_strategy
