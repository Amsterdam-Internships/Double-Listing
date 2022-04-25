import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import requests
import torchvision.transforms.functional as fn
from torchvision import transforms

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
    if type(string) == str:
        res = re.sub(r'[^\w\s]', ' ', string)
        res = res.lower()
        return res
    else:
        return np.nan
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

def clean_descrp(string):
    string = string.replace('<br />', '').replace('<b>', '').replace('</b>', '').replace('Registratienummer', '')
    string = string.replace('\x80', '').replace('\x92', '').replace('\x94', '').replace('\x91', '')
    string = re.sub('[0-9][0-9][0-9][0-9] [0-9][0-9][0-9][A-Z] [0-9][A-Z][0-9][0-9] [0-9][0-9][A-Z][0-9] [A-Z][0-9][A-Z][A-Z]','',string)
    string = re.sub('[0-9]','',string)
    
    string= clean_string(string)
    string =string.replace('br', '')
    return(string)
    

def join(strings):
    final_string = ''
    for x in strings:
        x = clean_descrp(x)
        if not re.match(x,final_string):
            final_string += x
            final_string += ' '
    return final_string
 
def retreive_img():
    return