# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


CLASSIFIERS = {
    "cat": "Cat",
    "dt": "DT",
    "xgb": "XGB",
    "lgbm": "LGBM",
    "svm": "SVM",
    "mlp": "MLP",
}

OVERSAMPLERS = {
    "none": "Baseline",
    "default": "Default",
    "random": "Random",
    "smote": "SMOTE",
    "svm": "SVM",
    "adasyn": "ADASYN",
    "border": "BL",
    "poly": "Poly",
}

METRICS = {
    "roc_auc": "AUC",
    "neg_brier_score": "Minus Brier score",
    "f1": "F1",
    "f2": "F2",
    "jaccard": "Jaccard",
    "balanced_accuracy": "Balanced_accuracy",
    "neg_log_loss": "Minus log-loss",
    "class_log_loss": "",
}


def filter_optimal_hps(df: pd.DataFrame, opt_metric: str, output_metrics: list):
    """
    For each {dataset, seed, oversampler, classifier} keep only the results of the HP configuration that
    yield the best score according to opt_metric. Then, calculate average and rank for the scores in
    output_metrics

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe of the experiment results
    opt_metric : str
        metric used for optimization
    output_metrics : list
        metrics to include in the output

    Returns
    -------
    dict: pandas.DataFrame
        Filtered and summarized
    """
    num_datasets = len(np.unique(df["dataset"]))
    num_seeds = len(np.unique(df["seed"]))
    # Filter best models according to opt_metric
    df = (
        df.sort_values("param_set", ascending=False)
        .sort_values([opt_metric], ascending=False, kind="stable")
        .groupby(["dataset", "seed", "learner", "oversampler"])
        .agg({om: "first" for om in output_metrics})
        .reset_index()
    )

    # Rank models per dataset and seed
    for om in output_metrics:
        df[f"{om}.rank"] = df.groupby(["dataset", "seed"])[om].rank(ascending=False)
    # Aggregate mean and rank over the datasets
    df = df.groupby(["learner", "seed", "oversampler"]).agg(
        {
            **{om: "mean" for om in output_metrics},
            **{f"{om}.rank": "mean" for om in output_metrics},
            "dataset": "count",
        }
    )
    # Aggregate mean and std over the seeds
    df = df.groupby(["learner", "oversampler"]).agg(
        {
            **{om: ["mean", "std"] for om in output_metrics},
            **{f"{om}.rank": ["mean", "std"] for om in output_metrics},
            "dataset": "sum",
        }
    )
    # Verify that all models have values for all datasets and seeds
    assert np.max(df["dataset"].to_numpy().ravel()) == num_datasets * num_seeds
    assert np.min(df["dataset"].to_numpy().ravel()) == num_datasets * num_seeds
    return df


def avg_plots(df: pd.DataFrame, metric: str, plot_rank: bool = True):
    """
    For each {dataset, seed, oversampler, classifier} keep only the results of the HP configuration that
    yield the best score according to opt_metric. Then, calculate average and rank for the scores in
    output_metrics

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered and summarized dataframe, produced by filter_optimal_hps
    metric : str
        metric to present
    plot_rank : bool
        Whether to plot rank or not

    Returns
    -------
    None
    """
    score_mean = []
    score_std = []
    rank_mean = []
    rank_std = []
    model_names = []
    major_ticks = []
    classifiers = list(np.unique(df.reset_index()["learner"]))
    oversamplers = list(np.unique(df.reset_index()["oversampler"]))
    for classifier in classifiers:
        for oversampler in oversamplers:
            idx = (classifier, oversampler)
            score_mean.append(df.loc[idx][(metric, "mean")])
            score_std.append(df.loc[idx][(metric, "std")])
            rank_mean.append(df.loc[idx][(f"{metric}.rank", "mean")])
            rank_std.append(df.loc[idx][(f"{metric}.rank", "std")])
            model_name = CLASSIFIERS[classifier]
            if oversampler != "none":
                model_name += "+" + OVERSAMPLERS[oversampler]
            model_names.append(model_name)
        # Add an empty row between classifiers
        score_mean.append(np.nan)
        score_std.append(np.nan)
        rank_mean.append(np.nan)
        rank_std.append(np.nan)
        model_names.append(" " * len(model_names))
        major_ticks.append(len(score_mean) - 1)

    # Delete the last empty row
    score_mean = score_mean[:-1]
    score_std = score_std[:-1]
    model_names = model_names[:-1]
    major_ticks = major_ticks[:-1]
    rank_mean = rank_mean[:-1]
    rank_std = rank_std[:-1]

    fig_height = 9 / (4 * 8) * (len(classifiers) * (len(oversamplers) + 1))
    plt.figure(figsize=(5, fig_height), dpi=320)
    ax = plt.axes()
    if plot_rank:
        ax2 = ax.twiny()
        ax2.errorbar(x=rank_mean, y=range(len(score_mean)), xerr=rank_std, fmt="r^")
        ax2.set_xlabel("Rank")
        ax2.xaxis.label.set_color("red")
        for t in ax2.xaxis.get_ticklabels():
            t.set_color("red")
    ax.errorbar(x=score_mean, y=range(len(score_mean)), xerr=score_std, fmt="bo")
    ax.xaxis.grid(True)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticklabels("" * len(major_ticks), minor=False)
    ax.set_yticks(range(len(model_names)), minor=True)
    ax.set_yticklabels(model_names, minor=True)
    ax.yaxis.grid(True, which="major")
    ax.set_xlabel(METRICS[metric.split(".")[1]])
    if plot_rank:
        ax.xaxis.label.set_color("blue")
        for t in ax.xaxis.get_ticklabels():
            t.set_color("blue")
    plt.show()
