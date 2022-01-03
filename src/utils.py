# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _process_dataset(x: pd.DataFrame, y: np.ndarray, normalize: bool):
    x.columns = x.columns.astype(str)
    transformers = []
    str_transform = [("ordinalEncoder", OrdinalEncoder())]
    numeric_transform = [("passthrough", "passthrough")]
    if normalize:
        str_transform.append(("StandardScaler", StandardScaler()))
        numeric_transform.append(("StandardScaler", StandardScaler()))
    for col in x.columns:
        pp = str_transform if x[col].dtype == "object" else numeric_transform
        transformers.append((col, Pipeline(pp), [col]))
    x = (
        ColumnTransformer(transformers, sparse_threshold=0)
        .fit_transform(x)
        .astype(float)
    )
    y = _encode_target(y)
    return x, y


def _encode_target(y: np.ndarray):
    # Encode the target column with the value 0 for the majority class and 1 for the minority class
    y = y.reshape((-1, 1))
    unique, counts = np.unique(y, return_counts=True)
    assert len(unique) == 2
    ret = np.zeros_like(y)
    if counts[0] > counts[1]:
        ret[y == unique[1]] = 1
    else:
        ret[y == unique[0]] = 1
    assert np.sum(ret == 0) == max(counts)
    assert np.sum(ret == 1) == min(counts)
    return ret
