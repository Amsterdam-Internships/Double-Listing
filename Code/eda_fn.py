import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def histogram(df,column,bins=False):
    #Histograms for EDA
    try:
        if df[column].dtype == 'object':
            df_1 = df[column].dropna().str.replace(',', '.').astype('float64')
        else:
            df_1 = df[column].dropna().astype('float64')
    except AttributeError:
        df_1 = df[column].astype('float64')
         
    bins = len(set(list(df[column].dropna())))
    if bins <= 25 :
        plt.hist(df_1,bins=bins)
        plt.title(column)
        plt.show()
    else:
        plt.hist(df_1)
        plt.title(column)
        plt.show()
    return 


def tot_nan_column(df,column):  
    df[column].notna().sum()
    nan_val= df[column].notna().sum()
    return nan_val

#Calculate non nan values in their columns
def in_nan(site,df_frame):
    nan_val = {}
    for x in df_frame.columns :   
            if df_frame[df_frame['site'] == site][x].notna().sum() != 0:
                    nan_val[x]= df_frame[df_frame['site'] == site][x].notna().sum()
                     
    return nan_val

def get_dataframe_info(df):
    """
    input
       df -> DataFrame
    output
       df_null_counts -> DataFrame Info (sorted)
    """

    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()
    
    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()
    
    # Reassign column names
    col_names = ["features", "types", "non_null_counts"]
    df_null_count.columns = col_names
    
    # Add this to sort
    
    
    return df_null_count

def clean_string(string):
    res = re.sub(r'[^\w\s]', '', string)
    res = res.lower()
    return res 
    
def helper_negative_ones(x):
    if x == -1 or x == '-1':
        return 1
    elif x == 0 or x == '0':
        return 0
    elif x == 'false':
        return 0
    elif x == 'true':
        return 1
    else:
        return x
 