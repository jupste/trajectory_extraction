import pandas as pd
import geopandas as gpd
import pyreadr
from pandas.api.types import is_numeric_dtype

def load_and_harmonize(filepath, columns_names= {'sourcemmsi':'mmsi', 'speedoverground':'speed'}):
    '''
    Load data from file and harmonize column names.

    Column names needed for further analysis:
    mmsi: int 
        the MMSI identifier
    lon: float 
        longitude parameter
    lat: float 
        latitude parameter
    status: int or sting 
        navigational status
    speed: float 
        ship speed
    t: int
        timestamp in UNIX epoch

    Parameters
    ----------
    filepath : path to csv or rds file
    column_names : dictionary that contains old and new column names in syntax {old: new}

    Returns
    -------
    df : pandas dataframe
    '''
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath) # config
    else:
        rdf = pyreadr.read_r(filepath)
        df = rdf[None].copy()
    static = pd.read_csv('../dbscan/lib/data/csv/static_data.csv')
    df.rename(columns = columns_names, inplace=True)
    vessel_types = static.groupby('sourcemmsi').shiptype.max()
    df['vessel_type'] = df.mmsi.map(vessel_types)
    df = df[df.vessel_type.isin(range(70,90))]
    if is_numeric_dtype(df['t']):
        df['date'] = pd.to_datetime(df.t, unit='s')
    else:
        df['date'] = df.t
    # 15 is AIS status for missing navigational status
    if not is_numeric_dtype(df['status']):
        df['status_name'] = df.status
        df.status = pd.factorize(df.status)[0]
    df.status.fillna(15,inplace=True)
    df = df.astype(dtype= {"mmsi":"int64",
        "lon":"float64","lat":"float64",
        "status":"int64",  "speed":"float64",
        "t": 'int64'})
    df.drop_duplicates(['mmsi', 't'], inplace=True)
    df.dropna(subset='heading', inplace = True)
    return df.sort_values(['mmsi', 't'])

