import pyreadr
import numpy as np
import pandas as pd
from data import Data, Points, Polygons
import config as cfg
from optimizer import Optimizer


if __name__ == '__main__':
    data = Data(path = cfg.DATA_PATH, column_names = cfg.COLUMN_NAMES)
    points = Points(data.data)
    optimizer = Optimizer(points.points)
    optimizer.generate_mooring_model()
    optimizer.generate_anchorage_model()
    points.add_anchorage_labels(optimizer.anchorage_model)
    points.add_mooring_labels(optimizer.mooring_model)
    polygons = Polygons(points.points)
    
    