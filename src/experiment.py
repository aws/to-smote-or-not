# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    fbeta_score,
    jaccard_score,
    balanced_accuracy_score,
    get_scorer,
)
from sklearn.model_selection import train_test_split

from classifiers import (
    _fit_xgboost,
    _fit_catboost,
    _fit_decision_tree,
    _fit_lgbm,
    _fit_svm,
    _fit_mlp,
)
from oversamplers import (
    _imblearn_oversample,
    _random_oversample,
    _poly_oversample,
)
from utils import _process_dataset


METRICS = [
    "roc_auc",
    "neg_brier_score",
    "f1",
    "f2",
    "jaccard",
    "balanced_accuracy",
    "neg_log_loss",
    "class_log_loss",
]


def experiment(
    x: pd.DataFrame,
    y: np.ndarray,
    oversampler: dict,
    classifier: dict,
    seed: int = 0,
    normalize: bool = False,
    clean_early_stopping: bool = False,
    consistent: bool = True,
    repeats: int = 1,
):
    """
    Run an experiment testing the performance of a classifier and an oversampler

    Parameters
    ----------
    x : pandas.DataFrame
        Feature data (could be raw. Doesn't have to be encoded)
    y : np.ndarray of size (-1, 1)
        Binary classification target column (could be raw. Doesn't have to be encoded)
    oversampler : a dict containing the keys:
        type: str in ["none", "default", "random", "smote", "svm", "adasyn", "border", "poly"]
        ratio: float. Desired imbalanced ratio. A value of 0.5 implies that the number of minority and majority samples
        is equal
        params: dict of oversampler HPs. See examples in OVERSAMPLER_HPS
    classifier : a dict containing the keys:
        type: str in ["cat", "dt", "xgb", "lgbm", "svm", "mlp"]
        params: dict of classifier HPs. See examples in CLASSIFIER_HPS
    seed : int
        random seed
    normalize: bool
        Whether to nomalize the data before oversampling
    clean_early_stopping: bool
        Whether to use two validation sets. One for early stopping and one for validation scores
    consistent: bool
        Whether to make the classifier consistent by optimizing the decision threshold on the validation data
    repeats: int
        number of train-validation folds to use

    Returns
    -------
    dict: metrics
    """

    x, y = _process_dataset(x, y, normalize)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=1 / 5, stratify=y, random_state=seed
    )
    results = []
    for repeat in range(repeats):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train_val,
            y_train_val,
            test_size=1 / 4,
            stratify=y_train_val,
            random_state=seed + repeat,
        )
        if clean_early_stopping:
            x_train, x_early_stopping, y_train, y_early_stopping = train_test_split(
                x_train,
                y_train,
                test_size=1 / 3,
                stratify=y_train,
                random_state=seed,
            )
        else:
            x_early_stopping = x_validation
            y_early_stopping = y_validation

        # oversample
        if oversampler["type"] in ["none", "default"]:
            data_valid = True
        elif oversampler["type"] in ["smote", "border", "svm", "adasyn"]:
            x_train, y_train = _imblearn_oversample(
                x=x_train,
                y=y_train,
                oversampler_name=oversampler["type"],
                params=oversampler["params"],
                seed=seed,
                ratio=oversampler["ratio"],
            )
            data_valid = x_train is not None
        elif oversampler["type"] == "random":
            x_train, y_train = _random_oversample(
                x=x_train,
                y=y_train,
                seed=seed,
                ratio=oversampler["ratio"],
            )
            data_valid = x_train is not None
        elif oversampler["type"] == "poly":
            x_train, y_train = _poly_oversample(
                x=x_train,
                y=y_train,
                params=oversampler["params"],
                seed=seed,
                ratio=oversampler["ratio"],
            )
            data_valid = x_train is not None
        else:
            raise Exception(f'ERROR: oversampler type = {oversampler["type"]}')

        if not data_valid:
            # balancing failed - don't return results
            return None

        model = {
            "xgb": _fit_xgboost,
            "cat": _fit_catboost,
            "dt": _fit_decision_tree,
            "lgbm": _fit_lgbm,
            "svm": _fit_svm,
            "mlp": _fit_mlp,
        }[classifier["type"]](
            x_train,
            y_train,
            x_early_stopping,
            y_early_stopping,
            seed,
            classifier["params"],
        )

        # calc metrics
        fold_results = {}
        for m in METRICS:
            fold_results.update(
                _calc_metric(
                    m,
                    model,
                    consistent,
                    (x_validation, y_validation),
                    (x_test, y_test),
                )
            )
        results.append(fold_results)

    # Average the results of repeating experiment
    keys_to_average = {}
    for k, v in results[0].items():
        if "threshold." in k or "test." in k or "validation." in k:
            keys_to_average[k] = []
    for r in results:
        for k, v in r.items():
            if k in keys_to_average.keys():
                keys_to_average[k].append(v)
    return {k: np.mean(v) for k, v in keys_to_average.items()}


def _calc_metric(
    metric: str,
    model,
    consistent: bool,
    validation_data: tuple,
    test_data: tuple,
):
    if metric in ["roc_auc", "neg_brier_score", "neg_log_loss"]:
        scorer = get_scorer(metric)
        return {
            f"validation.{metric}": scorer(
                model, validation_data[0], validation_data[1].ravel()
            ),
            f"test.{metric}": scorer(model, test_data[0], test_data[1].ravel()),
        }

    if metric == "class_log_loss":
        # calc neg_log_loss_0 and neg_log_loss_1
        d = {}
        for txt, data in [("validation", validation_data), ("test", test_data)]:
            proba = model.predict_proba(data[0])[:, 1]
            d[f"{txt}.log_loss_1"] = log_loss(
                data[1].ravel(), proba, sample_weight=data[1].ravel()
            )
            d[f"{txt}.log_loss_0"] = log_loss(
                data[1].ravel(), proba, sample_weight=(1 - data[1]).ravel()
            )
        return d

    val_proba = model.predict_proba(validation_data[0])[:, 1]
    val_y = validation_data[1].ravel()
    scorer, scorer_params = {
        "f1": (fbeta_score, {"beta": 1}),
        "f2": (fbeta_score, {"beta": 2}),
        "jaccard": (jaccard_score, {}),
        "balanced_accuracy": (balanced_accuracy_score, {}),
    }[metric]

    if consistent:
        best_validation_score = None
        best_validation_threshold = None
        for threshold in np.arange(0, 1, 0.01):
            pred = val_proba > threshold
            score = scorer(val_y, pred, **scorer_params)
            if not best_validation_score or score > best_validation_score:
                best_validation_threshold = threshold
                best_validation_score = score
    else:
        best_validation_threshold = 0.5
        pred = val_proba > best_validation_threshold
        best_validation_score = scorer(val_y, pred, **scorer_params)

    test_pred = model.predict_proba(test_data[0])[:, 1] > best_validation_threshold
    return {
        f"threshold.{metric}": best_validation_threshold,
        f"validation.{metric}": best_validation_score,
        f"test.{metric}": scorer(test_data[1].ravel(), test_pred, **scorer_params),
    }
