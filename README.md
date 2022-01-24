# To SMOTE, or not to SMOTE?
This package includes the code required to repeat the experiments in the paper and to analyze 
the results.

> To SMOTE, or not to SMOTE?
> 
> Yotam Elor and Hadar Averbuch-Elor

## Installation
```
# Create a new conda environment and activate it
conda create --name to-SMOTE-or-not -y python=3.7
conda activate to-SMOTE-or-not
# Install dependencies
pip install -r requirements.txt
```

## Running experiments
The data is not included with this package. See an example of running a single experiment with a dataset from 
`imblanaced-learn`
```python
# Load the data
import pandas as pd
import numpy as np
from imblearn.datasets import fetch_datasets
data = fetch_datasets()["mammography"]
x = pd.DataFrame(data["data"])
y = np.array(data["target"]).reshape((-1, 1))

# Run the experiment
from experiment import experiment
from classifiers import CLASSIFIER_HPS
from oversamplers import OVERSAMPLER_HPS
results = experiment(
    x=x,
    y=y,
    oversampler={
        "type": "smote",
        "ratio": 0.4,
        "params": OVERSAMPLER_HPS["smote"][0],
    },
    classifier={
        "type": "cat",  # Catboost
        "params": CLASSIFIER_HPS["cat"][0]
    },
    seed=0,
    normalize=False,
    clean_early_stopping=False,
    consistent=True,
    repeats=1
)

# Print the results nicely
import json
print(json.dumps(results, indent=4))
```
To run all the experiments in our study, wrap the above in loops, for example
```python
for dataset in datasets:
    x, y = load_dataset(dataset)  # this functionality is not provided
    for seed in range(7):
        for classifier, classifier_hp_configs in CLASSIFIER_HPS.items():
            for classifier_hp in classifier_hp_configs:
                for oversampler, oversampler_hp_configs in OVERSAMPLER_HPS.items():
                    for oversampler_hp in oversampler_hp_configs:
                        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                            results = experiment(
                                x=x,
                                y=y,
                                oversampler={
                                    "type": oversampler,
                                    "ratio": ratio,
                                    "params": oversampler_hp,
                                },
                                classifier={
                                    "type": classifier,
                                    "params": classifier_hp
                                },
                                seed=seed,
                                normalize=...,
                                clean_early_stopping=...,
                                consistent=...,
                                repeats=...
                            )
```
## Analyze
Read the results from the compressed csv file. As the results file is large, it is tracked using
[git-lfs](https://git-lfs.github.com/). You might need to download it manually or install git-lfs.
```python
import os
import pandas as pd
data_path = os.path.join(os.path.dirname(__file__), "../data/results.gz")
df = pd.read_csv(data_path)
```
Drop nans and filter experiments with consistent classifiers, no normalization and a single
validation fold
```python
df = df.dropna()
df = df[
    (df["consistent"] == True)
    & (df["normalize"] == False)
    & (df["clean_early_stopping"] == False)
    & (df["repeats"] == 1)
]
```

Select the best HP configurations according to AUC validation scores. `opt_metric` is the key
used to select the best configuration. For example, for a-priori HPs use `opt_metric="test.roc_auc"`
and for validation-HPs use `opt_metric="validation.roc_auc"`. Additionaly calculate average score and rank
```python
from analyze import filter_optimal_hps
df = filter_optimal_hps(
    df, opt_metric="validation.roc_auc", output_metrics=["test.roc_auc"]
)
print(df)
```
Plot the results
```python
from analyze import avg_plots
avg_plots(df, "test.roc_auc")
```
## Citation
```
@misc{elor2022smote,
    title={To SMOTE, or not to SMOTE?}, 
    author={Yotam Elor and Hadar Averbuch-Elor},
    year={2022},
    eprint={2201.08528},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

