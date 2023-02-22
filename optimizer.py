import pandas as pd
import numpy as np
import pygad
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import tqdm
import config as cfg
from data import get_crs
from sklearn.metrics import euclidean_distances
from shapely import geometry
import geopandas as gpd
from hdbscan.validity import validity_index


class Optimizer():
    def __init__(self, data):
        self.data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs = 4326)
        self.algorithm = DBSCAN() #cfg.CLUSTERING_ALGORITHM
        self.mooring_model = None
        self.anchorage_model = None
        self.optimizer_search_area = [range(5,100), range(10,800)] # config
        self.train = None

    def fit_to_genetical(self, fitness_func):
        '''
        Fit data to genetical algorithm.
        :param fitness_func: fitness function used in the optimizing process
        :return:
        '''
        num_genes=2
        num_generations=200
        with tqdm.tqdm(total=num_generations, desc="[Optimizing with genetic algorithm]") as pbar:
                    ga_instance = pygad.GA(num_generations=num_generations,
                                sol_per_pop=10,
                                num_parents_mating=5,
                                keep_parents=2,
                                num_genes=num_genes,
                                gene_space= self.optimizer_search_area,
                                init_range_high=2,
                                parent_selection_type='tournament',
                                init_range_low=2,
                                mutation_probability=0.5,
                                fitness_func= fitness_func,
                                stop_criteria='saturate_20',
                                suppress_warnings=True, 
                                save_best_solutions=True,
                                on_generation=lambda _: pbar.update(1))
                    ga_instance.run()
        return ga_instance

    def generate_mooring_model(self):
        '''
        Generates the mooring model for the class object. Calculates the custom distances and changes the coordinates to
        Universal Traverse Mercator projection
        :return:
        '''
        df = self.data[self.data.label==5].copy()
        df = df[~df.heading.isna()]
        df.to_crs(get_crs(*df.iloc[0][['lat','lon']].values), inplace=True)
        df['lat_utm'] = df.geometry.y
        df['lon_utm'] = df.geometry.x
        coords = df[['lat_utm', 'lon_utm']].values
        penalty_matrix = self._heading_penalty_matrix(df.heading.values)
        dist = euclidean_distances(coords)
        self.train = dist + penalty_matrix
        ga = self.fit_to_genetical(self.fitness_func_mooring())
        self.mooring_model = DBSCAN(eps = int(ga.best_solution()[0][1]), min_samples=int(ga.best_solution()[0][0]), metric='precomputed').fit(self.train)
    
    def generate_anchorage_model(self):
        '''
        Generates the anchorage model for the class object. Changes the coordinates to Universal Traverse Mercator
        projection
        :return:
        '''
        df = self.data[self.data.label==1].copy()
        df.to_crs(get_crs(*df.iloc[0][['lat','lon']].values), inplace=True)
        df['lat_utm'] = df.geometry.y
        df['lon_utm'] = df.geometry.x
        coords = df[['lat_utm', 'lon_utm']].values
        self.train = coords
        ga = self.fit_to_genetical(self.fitness_func_anchorage())
        self.anchorage_model = DBSCAN(eps = int(ga.best_solution()[0][1]), min_samples=int(ga.best_solution()[0][0]), metric='euclidean').fit(self.train)



    def fitness_func_anchorage(self):
        '''
        Fitness function for the anchorage model. Optimizes the DBCV score
        :return: fitness function
        '''
        train = self.train
        data = self.data[self.data.label == 1].copy()
        def fitness_function(solution, solution_idx):
            model = DBSCAN(min_samples=int(solution[0]), eps=float(solution[1]), metric='euclidean')
            model.fit(train)
            try:
                score = validity_index(self.train, model.labels_, metric = 'euclidean')
            except:
                return -99
            if np.isnan(score):
                return -99
            #ratio = self._get_area(data)
            return score #* ratio
        return fitness_function


    def fitness_func_mooring(self):
        train = self.train
        data = self.data[self.data.label == 5].copy()
        def fitness_function(solution, solution_idx):
            model = DBSCAN(min_samples=int(solution[0]), eps=float(solution[1]), metric='precomputed')
            model.fit(train)
            try:
                score = silhouette_score(self.train, model.labels_, metric = 'precomputed')
            except:
                return -99
            if np.isnan(score):
                return -99
            data['cluster'] = model.labels_
            ratio = self._get_length_width_ratio(data)
            return ratio * score
        return fitness_function


    def _get_area(self, df):
        clusters = df.copy()
        clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
        clusters.reset_index(drop=True, inplace=True)
        clusters = clusters[clusters.cluster!=-1]
        gb = clusters.groupby('cluster')
        areas = []
        for y in gb.groups:
            df0 = gb.get_group(y).copy()
            point_collection = geometry.MultiPoint(list(df0['geometry']))
            convex_hull_polygon = point_collection.convex_hull
            weight = len(df0)/len(clusters)
            areas.append(convex_hull_polygon.area*weight)
        return np.sum(areas)

    def _get_length_width_ratio(self, df):
        clusters = df.copy()
        clusters.sort_values(by=['cluster'], ascending=[True], inplace=True)
        clusters.reset_index(drop=True, inplace=True)
        clusters = clusters[clusters.cluster!=-1]
        gb = clusters.groupby('cluster')
        ratios = []
        for y in gb.groups:
            df0 = gb.get_group(y).copy()
            point_collection = geometry.MultiPoint(list(df0['geometry']))
            convex_hull_polygon = point_collection.convex_hull
            if not isinstance(convex_hull_polygon, geometry.Polygon):
                ratios.append(0)
                continue
            box = convex_hull_polygon.minimum_rotated_rectangle
            x, y = box.exterior.coords.xy
            edge_length = (geometry.Point(x[0], y[0]).distance(geometry.Point(x[1], y[1])), geometry.Point(x[1], y[1]).distance(geometry.Point(x[2], y[2])))
            length = max(edge_length)
            width = min(edge_length)
            if width<1:
                width = 1
            weight = len(df0)/len(clusters)
            ratios.append(weight*(length/width))
        return np.sum(ratios)

    def _absolute_angle_difference(self, target, source):
        a = target - source
        a = np.abs((a + 180) % 360 - 180)
        b = target - source - 180
        b = np.abs((b + 180) % 360 - 180)
        return min(a,b)

    def _heading_penalty_matrix(self, directions):
        dir_matrix = np.zeros([len(directions), len(directions)])
        for i in range(len(directions)):
            for j in range(len(directions)):
                if self._absolute_angle_difference(directions[i], directions[j])>15:
                    dir_matrix[i][j] = 10000
        return dir_matrix
    def _set_utm(self, data):
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs=4326)
        utm_zone = utm.from_latlon(*data.iloc[0][['lat', 'lon']].values)
        if utm_zone[3] > 'N':
            epsg = '326'
        else:
            epsg = '327'
        epsg = epsg + str(utm_zone[2])
        return data.to_crs(f'epsg:{epsg}')


