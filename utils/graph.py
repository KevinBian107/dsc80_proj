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

'''This file is for graphing only'''

# The stuff below is for Lecture 7/8.
def create_kde_plotly(df, group_col, group1, group2, vals_col, title=''):
    fig = ff.create_distplot(
        hist_data=[df.loc[df[group_col] == group1, vals_col], df.loc[df[group_col] == group2, vals_col]],
        group_labels=[group1, group2],
        show_rug=False, show_hist=False
    )
    return fig.update_layout(title=title)

def multiple_hists(df_map, histnorm="probability", title=""):
    values = [df_map[df_name]["child"].dropna() for df_name in df_map]
    all_sets = pd.concat(values, keys=list(df_map.keys()))
    all_sets = all_sets.reset_index()[["level_0", "child"]].rename(
        columns={"level_0": "dataset"}
    )
    fig = px.histogram(
        all_sets,
        color="dataset",
        x="child",
        barmode="overlay",
        histnorm=histnorm,
    )
    fig.update_layout(title=title)
    return fig


def multiple_kdes(df_map, title=""):
    values = [df_map[key]["child"].dropna() for key in df_map]
    labels = list(df_map.keys())
    fig = ff.create_distplot(
        hist_data=values,
        group_labels=labels,
        show_rug=False,
        show_hist=False,
        colors=px.colors.qualitative.Dark2[: len(df_map)],
    )
    return fig.update_layout(title=title).update_xaxes(title="child")