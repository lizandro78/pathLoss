import numpy as np
import pandas as pd
from GeoUtils import GeoUtils


class CdbFactory(object):
    """CdbFactory"""

    @staticmethod
    def CreateCdb(latMin, lonMin, latMax, lonMax, cellWidthInMts):
        """Generate a DataFrame with coordenates(lat and lon) for the Correlation DataBase"""
        grid_width = GeoUtils.distanceInKm(latMin, lonMin, latMin, lonMax)
        grid_height = GeoUtils.distanceInKm(latMin, lonMin, latMax, lonMin)
        ticks_hor = ((grid_width) // (cellWidthInMts / 1000))
        ticks_ver = ((grid_height) // (cellWidthInMts / 1000))
        lats = np.linspace(latMin, latMax, int(ticks_ver))
        lons = np.linspace(lonMin, lonMax, int(ticks_hor))
        grid = np.array(np.meshgrid(lats, lons)).reshape(2, -1).T
        df_grid = pd.DataFrame(grid, columns=["lat", "lon"])
        return df_grid
