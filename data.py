import pandas as pd
import geopandas as gpd
import utm 
from preprocess import load_and_harmonize
from scipy.stats import circstd, circmean
import numpy as np
from shapely.geometry import Point, MultiPoint

class Data():
    def __init__(self, path, columns_names):
        data = load_and_harmonize(path, columns_names) # config?
        self.data = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(data.lon, data.lat), crs = 4326)
        self._calculate_distances()

    def _calculate_distances(self):
        gdata = self.data.to_crs(get_crs(*self.data[['lon', 'lat']].values))
        self.data['distance'] = gdata.distance(gdata.shift())


    def include_polygons(self, polygons, labels):
        self.data[labels] = -1
        for _, cluster in polygons[1:].iterrows():
            geom = cluster.geometry
            sindex = self.data.sindex
            possible_matches_index = list(sindex.intersection(geom.bounds))
            possible_matches = self.data.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(geom)]
            self.data.loc[precise_matches.index, labels] = cluster.cluster_id

class Polygons():
    def __init__(self, data):
        self.anchorage_polygons = None
        self.mooring_polygons = None

    def set_polygons(self, moor_data, anch_data): 
        self.set_anchorage_polygons(anch_data)
        self.set_mooring_polygons(moor_data)

    def set_anchorage_polygons(self, data):
        self.anchorage_polygons = self.make_polygons(data)

    def set_mooring_polygons(self, data):
        self.mooring_polygons = self.make_polygons(data)

    def make_polygons(self, data):
        '''    
        Create convex hull polygons from points in clusters

        Parameters
        ----------
        clusters : DataFrame
            point dataframe with cluster membership
        Returns
        clusters : DataFrame
            dataframe with shapely polygons covering the area of clusters
        '''
        clusters = data.copy()
        clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
        clusters.reset_index(drop=True, inplace=True)
        clusters['geometry'] = [Point(xy) for xy in zip(clusters['lon'], clusters['lat'])]
        poly_clusters = gpd.GeoDataFrame()
        gb = clusters.groupby('cluster')
        for y in gb.groups:
            df0 = gb.get_group(y).copy()
            point_collection = MultiPoint(list(df0['geometry']))
            convex_hull_polygon = point_collection.convex_hull
            poly_clusters = poly_clusters.append(pd.DataFrame(data={'cluster_id':[y],'geometry':[convex_hull_polygon]}))
        poly_clusters.reset_index(inplace=True)
        poly_clusters.crs = 'epsg:4326'
        return gpd.GeoDataFrame(poly_clusters, crs='epsg:4326')

class Points():
    def __init__(self, data):
        self.speed_thresh = 0.5 # config
        self.time_thresh = 900 # config
        self.points = self._detect_anchorage_and_mooring(self._calculate_area_of_interest(data))
        self.anchorage_points = self.points[self.points.label==1]
        self.mooring_points = self._filter_by_harbor(self.points[self.points.label==5])
        
    def _calculate_area_of_interest(self, df):
        d = df.copy()
        # todo: halutaanko joku aika threshold myös?
        d['area_of_interest'] = (d.mmsi!=d.mmsi.shift()) | (d.distance > 200) | ((d.speed<self.speed_thresh) & (d.speed.shift()>=self.speed_thresh)) | ((d.speed>=self.speed_thresh) & (d.speed.shift()<self.speed_thresh))
        # mahdollisesti jos halutaan pelata vain nav numeilla
        # d['area_of_interest'] = (d.mmsi!=d.mmsi.shift()) | (d.status != d.status.shift())
        d['aoi_num'] = d.area_of_interest.cumsum()
        return d

    def _detect_anchorage_and_mooring(self, df):
        # Filter out only areas of interest where the speed stays low for 1 hour or more    
        df2 = df[df.speed<self.speed_thresh].groupby('aoi_num').filter(lambda x: x.t.max()-x.t.min() >= 3600)
        groups = df2.groupby('aoi_num')
        aoi_prediction = groups.apply(lambda x: 5 if np.degrees(circstd(np.radians(x.heading.values)))<2 or x.speed.mean()==0.00 else 1)
        df['predictive'] = df.aoi_num.map(aoi_prediction)
        g = df.groupby('aoi_num')
        coords = g[['lat', 'lon']].mean()
        coords['label'] = g.predictive.max()
        coords['heading'] = g.heading.apply(lambda x : np.degrees(circmean(x)))
        coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.lon, coords.lat), crs = 4326)
        return coords


    def _filter_by_harbor(self, gdf, include = True, radius = 5000):
        harbor_data = gpd.read_file('../dbscan/lib/data/shape/WPI.shp') # config
        c = harbor_data[harbor_data.PORT_NAME.str.contains('BREST')][['LONGITUDE', 'LATITUDE']].values
        crs = get_crs(*c[0])
        harbor_coords = gpd.GeoSeries([Point(xy) for xy in c], crs = 4326).to_crs(epsg= crs)
        if include: 
            return gdf[gdf.to_crs(epsg=crs).geometry.apply(lambda x: x.distance(harbor_coords[0]))<radius]
        return gdf[gdf.to_crs(epsg=crs).geometry.apply(lambda x: x.distance(harbor_coords[0]))>=radius] 

    def add_anchorage_labels(self, model):
        self.anchorage_points['cluster'] = model.labels_

    def add_mooring_labels(self, model):
        self.mooring_points['cluster'] = model.labels_


class Trajectories():
    def __init__(self, data):
        self.data = data
        self.aggregate_data = self._aggregate_points()
        self.trajectories = self._calculate_trajectories()


    def _aggregate_points(self):
        data = self.data
        points_of_interest = data[((data.anchorage != -1) & (data.status==1) | (data.mooring != -1) & (data.status==5)) ]
        points_of_interest['change'] = ((points_of_interest.anchorage != points_of_interest.anchorage.diff()) | points_of_interest.mooring != points_of_interest.mooring.diff() | points_of_interest.mmsi != points_of_interest.mmsi.diff())
        points_of_interest['event_number'] = points_of_interest['change'].cumsum()
        points_of_interest['next_moor'] = points_of_interest.mooring.shift(-1)
        return points_of_interest

    def _calculate_trajectories(self):
        gb = self.aggregate_data.groupby('event_number')
        trajectories_df = pd.DataFrame(columns = ['departure_anchorage', 'arrival_moor', 'shiptype', 'arrival_time']) 
        trajectories_df.departure_anchorage = gb.departure_anchorage.median()
        trajectories_df.arrival_moor = gb.next_moor.max()
        trajectories_df.shiptype = gb.shiptype.max()
        trajectories_df.arrival_time = gb.t.min()



def get_crs(lon, lat):
    utm_zone = utm.from_latlon(lat, lon)
    if utm_zone[3]>'N':
        epsg = '326'
    else:
        epsg = '327'
    return int(epsg + str(utm_zone[2]))