import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sklearn.preprocessing as pre

from typing import List, Union
from pathlib import Path
import joblib
import os


def get_country_df(country, icews_dir):
    """
    This function is meant to assist with loading ICEWS data from file. It first
    checks to see if there is a raw parquet file containing data from all countries,
    and if one does not exist it is created. It then filters and returns data from
    the country of choice.
    
    :param country: String. Country name of interest.
    :param icews_dir: The relative or absolute path to the folder containing both
        individual ICEWS tabular data files or a combined all country parquet
        file.
    """
    icews_dir = Path(icews_dir)
    
    country_filename = Path(f"icews_{country.lower().replace(' ', '_')}_raw.parquet")
    country_parquet_path = icews_dir / country_filename
    all_data_parquet_path = icews_dir / 'icews_all_raw.parquet'

    # Does country specific aggregated data exist already? If not...
    if not os.path.isfile(country_parquet_path):
        all_data_parquet = icews_dir / 'icews_all_raw.parquet'

        # Does un-aggregated data file already exist? If not...
        if not os.path.isfile(all_data_parquet_path):    
            icews_files = _get_file_list()
            df = None

            # read each year's tabular data and concat it to the dataframe:
            for f in icews_files:    
                file_path = icews_dir / f
                _clean_quotes(file_path)

                print('Creating dataframe from file... ', end=' ')
                tdf = pd.read_csv(file_path, sep='\t', engine='python',
                                  quotechar='"', error_bad_lines=False,
                                  converters={'CAMEO Code': lambda x: str(x)})
                                 
                print(f'done.\nshape: {tdf.shape}')
                if df is None:
                    df = tdf.copy()
                else:        
                    df = pd.concat([df, tdf])
                    df.index = range(df.shape[0])

            # Convert event dates to datetime type
            df['Event Date'] = pd.to_datetime(df['Event Date'].astype('string'))

            # Convert CAMEO codes from object to integer type:            
            df['CAMEO Code'] = df['CAMEO Code'].apply(_clean)
            
            # Create quad classes from cameo codes
            df['QuadClass'] = df['CAMEO Code'].apply(_quad_class)

            # save the resulting dataframe (2014 - 2020) to parquet file
            print(f'Saving {all_data_parquet} file...', end='')  
            df.to_parquet(all_data_parquet)
            print('done.')
            print(f'\nProcess complete. Combined dataframe shape: {df.shape}')
            
        # If all data parquet file does exist, load it:
        else:
            print(f'Loading {all_data_parquet_path} file...', end='')
            df = pd.read_parquet(all_data_parquet_path)
            print('done.')
            
        country_df = _select_country(df, country)
        
        if country_df is None:
            print('Country not found. None returned, nothing saved.')
            return None
        country_df.to_parquet(country_parquet_path)
        del df
     
    # If country data parquet file already exists, load it:
    else:
        print(f'loading {country_parquet_path}... ', end='')
        country_df = pd.read_parquet(country_parquet_path)
        print('done.')
        
    return country_df
  

def _get_file_list():
    """
    Helper function. Load list of individual ICEWS tabular data file names maintaned in 
    util/icews_file_list.txt. One file name per line. 
    """
    file_list = []
    with open('util/icews_file_list.txt', 'r') as f:
        for line in f.readlines():
            file_list.append(line.rstrip())
    return file_list

def _clean(x):
    """
    Helper function. CAMEO codes are twice dirty in ICEWS. First, many contain
    trailing letters that do not belong. Second,  many are missing leading zeros
    which are important place holders that signify catagories and sub-catagories.
    
    This method cleans up these issues and returns the top level category of the
    CAMEO code.
    """    
    # Check for and strip trailing letter if it exists
    try:
        int(x)
    except:
        x = str(x)[:-1]
    x = str(x)

    # Replace stripped leading zero if needed:
    if len(x) == 2:
        if int(x) > 20:
            return int('0' + x[0])
    if len(x) == 3:
        if int(x) > 200:
            return int('0' + x[0])

    return int(x[:2])
        
def _clean_quotes(file):
    """
    ICEWS data sometimes has unescaped, nested double quotes for actor nick-names. This
    causes Pandas to skip reading these line. Using this helper function to clean these 
    improper quotations will preserve more of the data during loading.
    """
    print(f'Searching {file} for illegal characters... ', end= ' ')
    with InPlace(file, encoding="ISO-8859-1") as f: 
        for line in f:
            line = line.replace('\"', '')
            line = line.replace('"', '')
            f.write(line)
    print('done.')
    
def _quad_class(x):
    """
    Convert ICEWS CAMEO codes to Quad Class.
    """
    cue_cat = int(str(x)[:2])
    if x < 6:
        return 2
    if x < 10:
        return 1
    if x < 15:
        return 3
    if x <= 20:
        return 4 
    print(x) # Should not execute

    
def _select_country(df, country):
    """
    Helper Function. Selects all rows of a specific country from given dataframe.
    Returns the country subset as it's own re-indexed dataframe.
    """
    idx = (df['Country']==country)
    country_df = df[idx]
    country_df.index = range(country_df.shape[0])  
    if country_df.empty:
        return None
    return country_df



def ohe_icews(df : pd.DataFrame, cats : List[str] = None) -> pd.DataFrame:
    """
    One-hot-encode desired columns of ICEWS dataframe.
    
    :param df: The dataframe to encode.
    :param cats: A list of ICEWS categories to encode. If none provided, only the 
        ``QuadClass`` will be encoded.
    :return df: Returns the dataframe with encoded columns. Original column is dropped.
    """
    
    ohe_qc = OneHotEncoder(sparse=False)

    if not cats:
        cats = ['QuadClass']     
    if len(cats) == 1:
        ohe_qc = ohe_qc.fit(df[cats].to_numpy().reshape(-1, 1))
    else:
        ohe_qc = ohe_qc.fit(df[cats])
        
    oh_cats = ohe_qc.transform(df[cats])

    # Get numpy one-hot encodings into a dataframe for concatenation: 
    columns = np.concatenate(ohe_qc.categories_).astype(str)
    oh_cats = pd.DataFrame(oh_cats, columns=columns, index=df.index)

    # concatenate encodings to original dataframe. Drop originals:
    df = pd.concat([df, oh_cats], axis=1)
    df = df.drop(cats, axis=1)
    df = df.reset_index(drop=True)
    
    return df

def to_timeseries(df : pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe of events data and creates a time series of average rates of present categories.
    """

    # Sort and group by date
    df = df.sort_values(by='Event Date')
    df = df.groupby('Event Date').mean() 
    
    return df

def scale_intensity(df : pd.DataFrame) -> pd.DataFrame:
    """
    Scales the "Intensity" variable to the [0, 1] range.
    """
    # Actual range of possible intensity values:
    in_min, in_max = -10, 10

    # Fit scaler based on actual max/min values
    scaler = pre.MinMaxScaler()
    scaler = scaler.fit([[in_min], [in_max]])

    # apply MinMaxScaler
    df.Intensity = scaler.transform(
        df.Intensity. \
        to_numpy(). \
        reshape(-1, 1)
    )
    
    return df


def run(df : pd.DataFrame, cats : List[str] = ['QuadClass'], path : Union[str, bytes, os.PathLike] = None) -> pd.DataFrame:
    """
    This function brings to gether a set of pre-processing steps specifically for ICEWS data.
    
    You pass the categories that you want to keep for one-hot encoding, and it drops everything else
    from the dataframe. What remains is one-hot encoded and then the whole dataframe is converted to
    timeseries format. 
    
    :param df: A dataframe containing raw ICEWS data. 
    :param cats: A list of categories you wish to keep. The QuadClass and Intensity variables are kept
        by default.
    :param path: An optional path argument. Your relative save-to location. 
    
    """

    # Drop everything except QuadClass and Intensity. 
    drop = set([
        'Event ID', 'ï»¿Event ID', 'Source Name', 'Source Sectors',
        'Source Country', 'CAMEO Code', 'Target Name', 'Target Sectors',
        'Target Country', 'Story ID', 'Sentence Number', 'Publisher',
        'City', 'District', 'Province', 'Latitude', 'Longitude',
        'Event Text', 'Country'
    ])

    # In case we later want to keep other categories
    drop = drop - (drop & set(cats))

    for d in drop:
        try:
            df = df.drop(d, axis=1)
        except:
            pass

    df.columns = df.columns.astype(str)
    df = ohe_icews(df, cats=cats)
    ts_df = to_timeseries(df)
    ts_df = scale_intensity(ts_df)
    
    if path is not None:
        ts_df.to_parquet(path)
    
    return ts_df
    