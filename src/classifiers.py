# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


CLASSIFIER_HPS = {
    "cat": [
        {},
        {"iterations": 500},
        {"boosting_type": "Ordered"},
        {"iterations": 500, "boosting_type": "Ordered"},
    ],
    "dt": [
        {"min_samples_leaf_factor": 0.005},
        {"min_samples_leaf_factor": 0.01},
        {"min_samples_leaf_factor": 0.02},
        {"min_samples_leaf_factor": 0.04},
    ],
    "xgb": [
        {"n_estimators": 1000, "learning_rate": 0.1},
        {"n_estimators": 1000, "learning_rate": 0.1, "subsample": 0.66},
        {"n_estimators": 1000, "learning_rate": 0.025},
        {"n_estimators": 1000, "learning_rate": 0.025, "subsample": 0.66},
    ],
    "lgbm": [
        {},
        {"subsample": 0.66, "subsample_freq": 1},
        {
            "learning_rate": 0.025,
            "n_estimators": 1000,
            "subsample": 0.66,
            "subsample_freq": 1,
        },
        {"learning_rate": 0.025, "n_estimators": 1000},
    ],
    "svm": [
        {"max_iter": 10000, "C": 1, "loss": "squared_hinge"},
        {"max_iter": 10000, "C": 1, "loss": "hinge"},
        {"max_iter": 10000, "C": 10, "loss": "squared_hinge"},
        {"max_iter": 10000, "C": 10, "loss": "hinge"},
    ],
    "mlp": [
        {"max_iter": 1000, "hidden_layer_sizes_ratio": 0.1, "activation": "relu"},
        {"max_iter": 1000, "hidden_layer_sizes_ratio": 0.5, "activation": "relu"},
        {"max_iter": 1000, "hidden_layer_sizes_ratio": 1.0, "activation": "relu"},
        {
            "max_iter": 1000,
            "hidden_layer_sizes_ratio": 0.5,
            "activation": "logistic",
        },
    ],
}


def _fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    class XGBClassifierWrapper(XGBClassifier):
        # A wrapper for XGBClassifier that automatically uses the best number of trees for inference when early
        # stopping is used in learning
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def predict(self, *args, **kwargs):
            return super().predict(
                *args, **kwargs, iteration_range=(0, self.best_iteration)
            )

        def predict_proba(self, *args, **kwargs):
            return super().predict_proba(
                *args, **kwargs, iteration_range=(0, self.best_iteration)
            )

    model = XGBClassifierWrapper(
        use_label_encoder=False,
        random_state=seed,
        n_jobs=1,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validattion)],
        eval_metric="logloss",
        early_stopping_rounds=10,
        verbose=False,
    )
    return model


def _fit_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    model = CatBoostClassifier(
        random_state=seed,
        thread_count=1,
        early_stopping_rounds=10,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validattion)],
        early_stopping_rounds=10,
        verbose=False,
    )
    return model


def _fit_decision_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    if "min_samples_leaf_factor" in params:
        params["min_samples_leaf"] = max(
            1, int(np.round(len(x_train) * params["min_samples_leaf_factor"]))
        )
        params.pop("min_samples_leaf_factor")
    model = DecisionTreeClassifier(random_state=seed, **params)
    model.fit(x_train, y_train)
    return model


def _fit_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    class LinearSVCWrapper(LinearSVC):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def predict_proba(self, X):
            pred = self.predict(X).reshape((-1, 1))
            proba = np.hstack([1 - pred, pred])
            return proba

    model = LinearSVCWrapper(random_state=seed, **params)
    model.fit(x_train, y_train.ravel())
    return model


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    if "hidden_layer_sizes_ratio" in params:
        params["hidden_layer_sizes"] = int(
            np.round(params["hidden_layer_sizes_ratio"] * x_train.shape[1])
        )
        params.pop("hidden_layer_sizes_ratio")
    model = MLPClassifier(random_state=seed, early_stopping=True, **params)
    model.fit(x_train, y_train)
    return model


def _fit_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validattion: np.ndarray,
    seed: int,
    params: dict,
):
    model = LGBMClassifier(
        random_state=seed,
        n_jobs=1,
        early_stopping_rounds=10,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validattion)],
        early_stopping_rounds=10,
        verbose=0,
    )
    return model
