import pyreadr
import numpy as np
import pandas as pd
from data import Data
from polygons import Polygons
import config as cfg


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = Data(path = cfg.DATA_PATH, column_names = cfg.COLUMN_NAMES)
    polygons = Polygons(data.data)
    