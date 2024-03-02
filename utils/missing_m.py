import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
from IPython.display import display, IFrame, HTML

import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"
from pathlib import Path

'''This file is for assesising missingness mechanisms'''

def multiple_describe(df_map):
    out = pd.DataFrame(
        columns=["Dataset", "Mean", "Standard Deviation"]
    ).set_index("Dataset")
    for key in df_map:
        out.loc[key] = df_map[key]["child"].apply(["mean", "std"]).to_numpy()
    return out

def make_mcar(data, col, pct=0.5):
    """Create MCAR from complete data"""
    missing = data.copy()
    idx = data.sample(frac=pct, replace=False).index
    missing.loc[idx, col] = np.NaN
    return missing


def make_mar_on_cat(data, col, dep_col, pct=0.5):
    """Create MAR from complete data. The dependency is
    created on dep_col, which is assumed to be categorical.
    This is only *one* of many ways to create MAR data.
    For the lecture examples only."""

    missing = data.copy()
    # pick one value to blank out a lot
    high_val = np.random.choice(missing[dep_col].unique())
    weights = missing[dep_col].apply(lambda x: 0.9 if x == high_val else 0.1)
    idx = data.sample(frac=pct, replace=False, weights=weights).index
    missing.loc[idx, col] = np.NaN

    return missing


def make_mar_on_num(data, col, dep_col, pct=0.5):
    """Create MAR from complete data. The dependency is
    created on dep_col, which is assumed to be numeric.
    This is only *one* of many ways to create MAR data.
    For the lecture examples only."""

    thresh = np.percentile(data[dep_col], 50)

    def blank_above_middle(val):
        if val >= thresh:
            return 0.75
        else:
            return 0.25

    missing = data.copy()
    weights = missing[dep_col].apply(blank_above_middle)
    idx = missing.sample(frac=pct, replace=False, weights=weights).index

    missing.loc[idx, col] = np.NaN
    return missing
