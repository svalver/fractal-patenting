
# Plotting codes for Figure 3 and 6 
# Created by Salva Duran-Nebreda (2024)
# Adapted by Sergi Valverde (2025)


# From the paper:  
# "Fractal clusters and urban scaling shape spatial inequality in U.S. patenting" 
# published in npj Complexity
# https://doi.org/10.1038/s44260-025-00054-y

# Authors:
# Salva Duran-Nebreda, Blai Vidiella,  R. Alexander Bentley and Sergi Valverde



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from collections import defaultdict

# the following import may require "conda install -c conda-forge basemap"
# see: https://www.geeksforgeeks.org/python/how-to-fix-the-no-module-named-mpl_toolkits-basemap-error-in-python/
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import os
from inequality import Point

def newyork_bbox (): 
    llcrnrlon = -80   # Lower-left longitude
    llcrnrlat = 39    # Lower-left latitude
    urcrnrlon = -70   # Upper-right longitude
    urcrnrlat = 46    # Upper-right latitude
    return [ llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat ]

def california_bbox():
    return [-125, 32, -113, 42]

def texas_bbox():
    """
    Returns the bounding box for the state of Texas.

    :return: (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    """
    return (-107.0, 25.5, -93.5, 37.0)

def new_mexico_bbox():
    """
    Returns the bounding box for the state of New Mexico.

    :return: (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    """
    return (-109.1, 31.2, -102.9, 37.1)

def new_jersey_bbox():
    """
    Returns the bounding box for the state of New Jersey.

    :return: (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    """
    return (-75.6, 38.8, -73.8, 41.4)

class CityMapPlotter:

    def __init__(self, points, data_folder, bounding_box=(-119, 20, -64, 49), figsize=(6, 4.5)):
        """
        :param points: List of (x, y) tuples or Point objects
        :param data_folder: Path to folder containing shapefile (e.g., 'st99_d00.shp')
        :param figsize: Tuple of figure size in inches
        """
        self.points_array = [Point(*pt) for pt in points]
        self.data_folder = data_folder
        self.figsize = figsize
        self.city_dict = defaultdict(int)
        self.map = None
        self.fig = None
        self.ax = None
        self.bounding_box = bounding_box
        # Colors
        self.light_gray = [0.8] * 3
        self.dark_gray = [0.6] * 3
        self.draw_states = True

    def create_map(self):
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = self.bounding_box
        self.map = Basemap(
            llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
            projection='lcc', lat_1=33, lat_2=45, lon_0=(llcrnrlon + urcrnrlon) / 2,
            resolution='i'
        )
        self.map.readshapefile(os.path.join(self.data_folder, 'st99_d00'), 'states',
                               drawbounds=True, linewidth=0.2, color=self.light_gray)
        # self.map.drawcountries(linewidth=0.9, linestyle='solid', color=self.dark_gray, ax=self.ax)
        self.map.drawcoastlines(linewidth=0.9, linestyle='solid', color=self.dark_gray, ax=self.ax)
        # 🔽 Disable axes box and ticks
        self.ax.set_frame_on(False)
        self.ax.axis('off')

    def create_map_solid(self):
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = self.bounding_box
        self.map = Basemap(
            llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
            projection='lcc', lat_1=33, lat_2=45,
            lon_0=(llcrnrlon + urcrnrlon) / 2,
            resolution='i'
        )

        # Fill land and sea
        self.map.drawmapboundary(fill_color='lightblue', ax=self.ax)
        # self.map.fillcontinents(color='white', lake_color='lightblue', ax=self.ax)

        if self.draw_states:
            # Optional: draw state boundaries (but no coastlines or countries)
            self.map.readshapefile(
                os.path.join(self.data_folder, 'st99_d00'),
                'states',
                drawbounds=True,
                linewidth=0.9,
                color=self.light_gray
            )

        # Disable axes box and ticks
        self.ax.set_frame_on(False)
        self.ax.axis('off')


    def process_city_data(self):
        self.city_dict = defaultdict(int)
        for point in self.points_array:
            self.city_dict[point] += 1

    def plot_solid_cities(self, city_scale = 20.0, cmap = "flare", min_count = 0.0):
        sorted_cities = sorted(self.city_dict.items(), key=lambda item: item[1])
        lats, lons, intensity, sizes = [], [], [], []
        for city, count in sorted_cities:
            if count >= min_count: 
                lons.append(city.x)
                lats.append(city.y)
                log_count = np.log10(count-min_count )
                intensity.append(log_count)
                sizes.append(city_scale * log_count)
        self.map.scatter(
            lons, lats, c=intensity, s=sizes,
            cmap=sns.color_palette(cmap, as_cmap=True),
            latlon=True, zorder=3
        )

    def plot_dot_cities(self, city_scale = 20.0, color= "gray"):
        sorted_cities = sorted(self.city_dict.items(), key=lambda item: item[1])
        lats, lons, intensity, [], [], [], []
        for city, count in sorted_cities:
            lons.append(city.x)
            lats.append(city.y)
            log_count = np.log10(count)
        self.map.scatter(
            lons, lats, c=color, s=city_scale,
            latlon=True, zorder=3
        )    

    def plot(self, city_scale = 20.0):
        # self.create_map()
        self.create_map_solid()
        self.process_city_data()
        self.plot_solid_cities(city_scale)


