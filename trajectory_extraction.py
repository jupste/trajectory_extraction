import pandas as pd
import geopandas as gpd
import numpy as np


def add_clusters_to_data(gdf, polygons, cluster_type):
    '''
    Adds cluster membership to each row in an AIS dataframe

    Parameters
    ----------
    gdf : GeoDataFrame
        geodataframe containing AIS messages
    polygons : GeoDataFrame
        clusters to be added to the AIS data
    cluster_type : string
        type of cluster
    Returns
    clusters : GeoDataFrame
        AIS dataframe with cluster membership
    '''
    gdf[cluster_type] = -1
    for _, cluster in polygons[1:].iterrows():
        geom = cluster.geometry
        sindex = gdf.sindex
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(geom)]
        gdf.loc[precise_matches.index, cluster_type] = cluster.cluster_id
    return gdf


def extract_trajectories(df):
    '''
    Extract trajectories between anchorages and mooring polygons from the data. Trajectory is defined as a linkage
    between an anchorage cluster and a mooring cluster.


    :param df: AIS dataframe with both mooring and anchorage cluster memberships attached
    :return: trajectory: dataframe that has all the trajectories. Note all columns are transferred from df to trajectory
    e.g. ship length information
    '''
    data = df.copy()
    data.sort_values(['id', 't'],inplace=True)
    # New are of interest if the moor changes or the anchorage changes or a new ship
    data['change'] = ((data.moor.diff() != 0) | ((data.anchorage.diff() != 0)) | (data.id.diff() != 0))
    data['change_num'] = data.change.cumsum()
    # Only groups which have 1 or 5 status in them
    g = data.groupby('change_num').status.apply(lambda x: (set([1]).issubset(x)) or (set([5]).issubset(x)))
    # Alternative only count events where the ship is in the cluster for at least a certain timeperiod
    # g = data.groupby('change_num').time.apply(lambda x: (x.max() - x.min())>3600)
    data = data[data.change_num.isin(g[g == True].index)]
    t = data.groupby(['change_num']).first()
    t = t[(t.moor != -1) | (t.anchorage != -1)]
    t['next_anch'] = t.anchorage.shift(-1)
    t['next_moor'] = t.moor.shift(-1)
    trajectory = t[((t.anchorage != -1) & (t.next_moor > -1) & (t.id == t.id.shift(-1)))]
    return trajectory
