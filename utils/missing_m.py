import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
from IPython.display import display, IFrame, HTML

from pathlib import Path
from scipy.stats import ks_2samp
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"

from utils.graph import *

'''This file is for assesising missingness mechanisms'''


def mar_making(data, col, dep_col, pct=0.5):
    """Create MAR from complete data. The dependency is
    created on dep_col, which is assumed to be categorical (contributor_id)."""

    missing = data.copy()
    # pick one value to blank out a lot
    high_val = np.random.choice(missing[dep_col].unique())
    weights = missing[dep_col].apply(lambda x: 0.9 if x == high_val else 0.1)
    idx = data.sample(frac=pct, replace=False, weights=weights).index
    missing.loc[idx, col] = np.NaN
    return missing


def mar_check_continuous(df,miss_col, dep_col):
    '''Full checking mar by simulating mar data then graphing it,
    miss_col must be catagorical and dep_col must be continuous'''
    # df_mar = df.copy()
    # df_mar = df_mar[[miss_col, dep_col]].dropna()
    # mar_missing = mar_making(df, miss_col, dep_col, pct=0.5)[miss_col].isna()
    # df_mar = df_mar.assign(mar_missing = mar_missing)
    
    missing = df[miss_col].isna()
    df_missing = df.assign(mar_missing = missing)[['mar_missing', dep_col]]

    fig = create_kde_plotly(df_missing, 'mar_missing', True, False, dep_col, title=f'MAR Graph of {miss_col} Dependent on {dep_col}')

    return fig.show()



def permutation_mean(df, miss_col, dep_col, rep):
    '''conduct permutation testing for testing mar in data frame '''

    # def mar(df, miss_col, dep_col):
    #     '''Generate mar column dataframe'''
    #     df_mar = df.copy()
    #     df_mar = df_mar[[miss_col, dep_col]].dropna()
    #     mar_missing = mar_making(df, miss_col, dep_col, pct=0.5)[miss_col].isna()
    #     df_mar = df_mar.assign(mar_missing = mar_missing)
    #     return df_mar
    
    def permutation_test(df, rep, dep_col):
        '''test_statistics is differences in the mean of quanitative column after grouping by True/False for missing value'''
        
        # line of missing of description that may base on dep_col?
        observe = df.groupby('mar_missing').mean()[dep_col].abs().diff().iloc[-1]
        
        # making a distrbution where missing of description does not depend on dep_col
        n_repetitions = rep
        null = []
        for _ in range(n_repetitions):
            with_shuffled = df.assign(shuffle = np.random.permutation(df['mar_missing']))
            group_means = (with_shuffled.groupby('shuffle').mean())['mar_missing']
            difference = group_means.diff().abs().iloc[-1]
            null.append(difference)
        return observe, null

    missing = df[miss_col].isna()
    df_missing = df.assign(mar_missing = missing)[['mar_missing', dep_col]]

    #mar_df = mar(df, miss_col, dep_col)
    observe, null = permutation_test(df_missing, rep, dep_col)

    fig = px.histogram(pd.DataFrame(null), x=0, histnorm='probability', title=f'Distribution for Null {miss_col}_col is dependent on {dep_col}_col')
    fig.add_vline(x=observe, line_color='red', line_width=1, opacity=1)

    p = (observe >= null).mean()
    print(f'p_value is {p}')
    
    return fig.show()


def permutation_ks(df, miss_col, dep_col, rep):
    '''conduct permutation testing for testing mar in data frame '''

    def permutation_test(df, rep, dep_col):
        '''test_statistics is the KS statistics'''
        
        # line of missing of description that may base on dep_col?
        observe = ks_2samp(df_missing.query('mar_missing')[dep_col],
                           df_missing.query('not mar_missing')[dep_col]).statistic
        
        # making a distrbution where missing of description does not depend on dep_col
        n_repetitions = rep
        null = []
        for _ in range(n_repetitions):
            with_shuffled = df.assign(shuffle = np.random.permutation(df['mar_missing']))
            difference = ks_2samp(with_shuffled.query('shuffle')[dep_col],
                                  with_shuffled.query('not shuffle')[dep_col]).statistic
            null.append(difference)
        return observe, null

    missing = df[miss_col].isna()
    df_missing = df.assign(mar_missing = missing)[['mar_missing', dep_col]]

    observe, null = permutation_test(df_missing, rep, dep_col)

    fig = px.histogram(pd.DataFrame(null), x=0, histnorm='probability', title=f'KS Distribution for Null {miss_col}_col is dependent on {dep_col}_col')
    fig.add_vline(x=observe, line_color='red', line_width=1, opacity=1)

    p = (observe <= null).mean()
    print(f'p_value is {p}')
    
    return fig.show()
