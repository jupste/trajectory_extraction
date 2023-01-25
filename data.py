import pandas as pd 
import numpy as np
from preprocess import load_and_harmonize
import geopandas as gpd

class Data():
    def __init__(self, path, columns_names):
        data = load_and_harmonize(path, columns_names) # config?
        self.data = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(data.lon, data.lat), crs = 4326)

    def include_polygons(self, polygons, labels):
        self.data[labels] = -1
        for _, cluster in polygons[1:].iterrows():
            geom = cluster.geometry
            sindex = self.data.sindex
            possible_matches_index = list(sindex.intersection(geom.bounds))
            possible_matches = self.data.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(geom)]
            self.data.loc[precise_matches.index, labels] = cluster.cluster_id