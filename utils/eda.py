import pandas as pd
from itertools import chain
import numpy as np
from pathlib import Path
import plotly.express as px
pd.options.plotting.backend = 'plotly'

'''Functions for transformation and data cleaning'''

def initial(df):
    '''Initial claeaning and megrging of two df, add average ratings'''

    # fillna 0 so avearge rating actually make sense
    df['rating'] = df['rating'].fillna(0)

    # not unique recipe_id
    avg = df.groupby('recipe_id')[['rating']].mean().rename(columns={'rating':'avg_rating'})
    df = df.merge(avg, how='left', left_on='recipe_id',right_index=True)
    return df


def transform_df(df):
    '''Transforming nutrition to each of its own catagory,
    tags, steps, ingredients to list,
    submission date to timestamp object,
    convert types,
    and remove 'nan' to np.NaN'''

    # Convert nutrition to its own caatgory
    data = df['nutrition'].str.strip('[]').str.split(',').to_list()
    name = {0:'calories',1:'total_fat',2:'sugar',3:'sodium',4:'protein',5:'sat_fat',6:'carbs'}
    #zipped = data.apply(lambda x: list(zip(name, x)))
    new = pd.DataFrame(data).rename(columns=name)

    df = df.merge(new,how='inner',right_index=True, left_index=True)
    df = df.drop(columns=['nutrition'])

    # Convert to list
    def convert_to_list(text):
        return text.strip('[]').replace("'",'').split(', ')
    
    df['tags'] = df['tags'].apply(lambda x: convert_to_list(x))
    df['ingredients'] = df['ingredients'].apply(lambda x: convert_to_list(x))

    # it's correct, just some are long sentences, doesn't see "'", notice spelling
    df['steps'] = df['steps'].apply(lambda x: convert_to_list(x)) #some white space need to be handled

    # submission date to time stamp object
    format ='%Y-%m-%d'
    df['submitted'] = pd.to_datetime(df['submitted'],format=format)
    df['date'] = pd.to_datetime(df['date'],format=format)

    # drop not needed & rename
    df = df.drop(columns=['id']).rename(columns={'submitted':'recipe_date','date':'review_date'})

    # Convert data type
    df[['calories','total_fat','sugar',
        'sodium','protein','sat_fat','carbs']] = df[['calories','total_fat','sugar',
                                                     'sodium','protein','sat_fat','carbs']].astype(float)
    df[['user_id','recipe_id','contributor_id']] = df[['user_id','recipe_id','contributor_id']].astype(str)

    df['rating'] = df['rating'].astype(int)

    # there are 'nan' values, remove that
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].apply(lambda x: np.NaN if x=='nan' else x)

    return df


def outlier(df):
    '''take care of outliers in the data frame'''

    check = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
    for col in check:#df.select_dtypes(include='number'):
        q_low = df[col].quantile(0.2)
        q_hi  = df[col].quantile(0.8)
        df_filtered = df[(df[col]<q_hi) & (df[col]>q_low)]

    return df_filtered


def norm(df):
    '''for standarlization of the numerical values in thed ata frame'''
    for col in df.select_dtypes(include='number').columns: 
        df[col] = df[col]/df[col].abs().max()
    return df


def group_recipe(df):
    def helper(series): # this runs slow
        return series.mean() if not isinstance(series, str) else series.first

    check_dict = {'minutes':'mean', 'n_steps':'mean', 'n_ingredients':'mean',
                'avg_rating':'mean', 'rating':'mean', 'calories':'mean',
                'total_fat':'mean', 'sugar':'mean', 'sodium':'mean',
                'protein':'mean', 'sat_fat':'mean', 'carbs':'mean',
                'steps':'first', 'name':'first', 'description':'first',
                'ingredients':'first', 'user_id':'first', 'contributor_id':'first',
                'review_date':'first', 'review':'first',  'recipe_date':'first',
                'tags':'first'}

    grouped = df.groupby('recipe_id').agg(check_dict)
    grouped['rating'] = grouped['rating'].astype(int)

    return grouped


def group_user(df):
    '''function for grouping by unique user_id and concating all steps/names/tags of recipe and averaging rating give'''
    
    return df.groupby('user_id')['steps','rating','name','tags'].agg({'steps':lambda x: list(chain.from_iterable(x)),
                                                        'name':lambda x: list(x),
                                                        'tags':lambda x: list(chain.from_iterable(x)),
                                                        'rating':'mean'})