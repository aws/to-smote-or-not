import os
import pandas as pd
import numpy as np

from analyze import filter_optimal_hps, avg_plots
from classifiers import CLASSIFIER_HPS
from oversamplers import OVERSAMPLER_HPS
from experiment import experiment


def test_experiments():
    # run all experiments on a single dataset
    x = pd.DataFrame(np.random.rand(100, 7))
    y = np.random.rand(100, 1) > 0.9

    for classifier, classifier_hp_configs in CLASSIFIER_HPS.items():
        for classifier_hp in classifier_hp_configs[:1]:
            for oversampler, oversampler_hp_configs in OVERSAMPLER_HPS.items():
                for oversampler_hp in oversampler_hp_configs[:1]:
                    print(f"Running experiment {classifier} - {oversampler}")
                    results = experiment(
                        x=x,
                        y=y,
                        oversampler={
                            "type": oversampler,
                            "ratio": 0.5,
                            "params": oversampler_hp,
                        },
                        classifier={"type": classifier, "params": classifier_hp},
                        seed=0,
                    )


def test_analyze():
    data_path = os.path.join(os.path.dirname(__file__), "../data/results.gz")
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df[
        (df["consistent"] == True)
        & (df["normalize"] == False)
        & (df["clean_early_stopping"] == False)
        & (df["repeats"] == 1)
    ]
    df = filter_optimal_hps(
        df, opt_metric="validation.roc_auc", output_metrics=["test.roc_auc"]
    )
    avg_plots(df, "test.roc_auc")
