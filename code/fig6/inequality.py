# Compute the multiscale coefficient of variation (CV(R)) for spatial data
# Evolution of Networks Lab
# Sergi Valverde, 2025
# @svalver
 

# From the paper:  
# "Fractal clusters and urban scaling shape spatial inequality in U.S. patenting" 
# published in npj Complexity
# https://doi.org/10.1038/s44260-025-00054-y

# Authors:
# Salva Duran-Nebreda, Blai Vidiella,  R. Alexander Bentley and Sergi Valverde



import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from math import sqrt
from collections import defaultdict, Counter
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import csv
import networkx as nx
# from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm.notebook import tqdm


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))  # rounded for floating-point stability

    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"
    
@dataclass
class Rectangle:
    x: float  # Lower-left X
    y: float  # Lower-left Y
    width: float
    height: float

    def contains(self, point: Point) -> bool:
        return (self.x <= point.x < self.x + self.width) and \
               (self.y <= point.y < self.y + self.height)

    def overlaps(self, other: 'Rectangle') -> bool:
        return not (
            self.x + self.width < other.x or
            other.x + other.width < self.x or
            self.y + self.height < other.y or
            other.y + other.height < self.y
        )

@dataclass
class QuadTreeItem:
    point: Point
    index: int

class QuadTreeNode:
    def __init__(self, boundary: Rectangle, level=0):
        self.boundary = boundary
        self.level = level
        self.children: List[Optional[QuadTreeNode]] = [None]*4
        self.items: List[QuadTreeItem] = []
        self.divided = False

    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.width / 2, self.boundary.height / 2
        self.children[0] = QuadTreeNode(Rectangle(x, y + h, w, h), self.level + 1)  # Top-left
        self.children[1] = QuadTreeNode(Rectangle(x + w, y + h, w, h), self.level + 1)  # Top-right
        self.children[2] = QuadTreeNode(Rectangle(x + w, y, w, h), self.level + 1)  # Bottom-right
        self.children[3] = QuadTreeNode(Rectangle(x, y, w, h), self.level + 1)  # Bottom-left
        self.divided = True

    def insert(self, point: Point, index: int, min_size=0.01):
        if not self.boundary.contains(point):
            return False

        if not self.divided and self.boundary.width > min_size:
            self.subdivide()
            for item in self.items:
                self._insert_into_children(item)
            self.items = []

        if self.divided:
            return self._insert_into_children(QuadTreeItem(point, index))
        else:
            self.items.append(QuadTreeItem(point=point , index=index))
            return True

    def _insert_into_children(self, item: QuadTreeItem) -> bool:
        for child in self.children:
            if child.insert(item.point, item.index):
                return True
        return False

    def query_circle(self, center: Point, radius: float, stop_radius: float = 0.0) -> List[QuadTreeItem]:
        found = []
        self._query_circle_recursive(center, radius, stop_radius, found)
        return found

    def _query_circle_recursive(self, center: Point, radius: float, stop_radius: float, found: List[QuadTreeItem]):
        bbox = Rectangle(center.x - radius, center.y - radius, 2 * radius, 2 * radius)
        if not self.boundary.overlaps(bbox):
            return

        dx = self.boundary.width * 0.5
        dy = self.boundary.height * 0.5
        diag_sq = (dx * dx) + (dy * dy)
        if diag_sq <= (stop_radius * stop_radius):
            return
        
        if self.divided:
            for child in self.children:
                child._query_circle_recursive(center, radius, stop_radius, found)
        else:
            for item in self.items:
                # Check if the point is within the circle
                # print(type(item.point))
                if center.distance_to(item.point) < radius:
                    found.append(item)

    def draw(self, ax):
        # Draw this node's rectangle
        rect = plt.Rectangle(
            (self.boundary.x, self.boundary.y),
            self.boundary.width,
            self.boundary.height,
            fill=False,
            edgecolor='gray',
            linewidth=0.5
        )
        ax.add_patch(rect)

        # Recurse to draw children
        if self.divided:
            for child in self.children:
                child.draw(ax)

    def count_items(self) -> int:
        if self.divided:
            return sum(child.count_items() for child in self.children)
        return len(self.items)

# === Helper Functions ===

def get_connected_components(graph: nx.DiGraph):
    # Convert to undirected to compute connected components
    return list(nx.connected_components(graph.to_undirected()))

def logspace(start, end, num):
    # Return log-spaced values between start and end
    return list(np.logspace(np.log10(start), np.log10(end), num))

def mean(values):
    return float(np.mean(values))

def var(values):
    return float(np.var(values))

# === Abstract Base Clustering Class ===

class SpatialClustering(ABC):
    class Observer(ABC):
        @abstractmethod
        def on_component_distribution(self, radius, patents_per_component):
            pass

    def __init__(self):
        self.minx = +1e6
        self.miny = +1e6
        self.maxx = -1e6
        self.maxy = -1e6

    def get_bbox(self, points):
        coords = np.array([(p.x, p.y) for p in points])
        self.minx, self.miny = np.min(coords, axis=0)
        self.maxx, self.maxy = np.max(coords, axis=0)
        print(f"minx={self.minx}\nmaxx={self.maxx}\nminy={self.miny}\nmaxy={self.maxy}")

    def run(self, points, radii, obs=None):
        self.init()
        self.get_bbox(points)

        # Define padded bounding box
        padding = 1e-6
        boundary = Rectangle(self.minx - padding, self.miny - padding,
                             (self.maxx - self.minx) + 2 * padding,
                             (self.maxy - self.miny) + 2 * padding)

        # Build quadtree and graph
        tree = QuadTreeNode(boundary)
        G = nx.DiGraph()
        for i, pt in enumerate(points):
            tree.insert(pt, i)
            G.add_node(i, pos=(pt.x, pt.y))

        print(f"Inserted {len(points)} points into quadtree")
        print(f"Inserted {G.number_of_nodes()} points into graph")
      
        # Process connectivity at each scale sequentially
        for radius in tqdm(radii, desc="Processing radii"):
            edge_count_before = G.number_of_edges()
            for i, pt in enumerate(points):
                neighbors = tree.query_circle(pt, radius)
                for qti in neighbors:
                    idx = qti.index
                    if i != idx and not G.has_edge(i, idx):
                        G.add_edge(i, idx)
            new_edges = G.number_of_edges() - edge_count_before
            self.process_groups(G, radius, obs)        
        self.done()

# === Concrete Subclass: Variance-Based Clustering ===

class VarianceClusters(SpatialClustering):
    def __init__(self, filename: str):
        super().__init__()
        self.output_file = filename
        self.patent_count: List[float] = []
        self.csv_file = None
        self.csv_writer = None

    def init(self):
        print("Initializing VarianceClusters...")
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["scale", "ncc", "meanpat", "stdpat", "nlinks", "fclust"])

    def done(self):
        print("Closing VarianceClusters output.")
        if self.csv_file:
            self.csv_file.close()

    def calculate_component_patents(self, components, patents_per_node):
        # For each component, sum the number of patents at each node
        patents_per_component = []
        for component_nodes in components.values():
            total = sum(patents_per_node[i] for i in component_nodes)
            patents_per_component.append(total)
        return patents_per_component

    def process_groups(self, graph, radius, obs=None):
        # Compute connected components
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        clustering_fraction = len(components) / graph.number_of_nodes()

        # Turn component list into dictionary
        component_dict = {i: list(comp) for i, comp in enumerate(components)}
        patents_per_component = self.calculate_component_patents(component_dict, self.patent_count)

        # Output row to CSV
        self.csv_writer.writerow([
            radius,
            len(components),
            mean(patents_per_component),
            sqrt(var(patents_per_component)),
            graph.number_of_edges(),
            clustering_fraction
        ])

        # Call optional observer hook
        if obs:
            obs.onComponentDistribution(radius, patents_per_component)

# Step 4: Define radii (log-spaced)
# g_logspace_min = -2
# g_logspace_max = 0.5
# g_logspace_steps = 32
# g_radii = logspace(10**2, 10**0.5, 32)

def run_variance_clustering(points_array: List[Point], 
                            radii: List[float], output_file: str):
    
    variance = VarianceClusters(output_file)
   
   # Step 2: Deduplicate points and count frequencies
    node_index = {}
    node_coords = []
    point_counts = defaultdict(int)

    for tuple in points_array:  # From extract_valid_points()
        pt = Point(*tuple)
        if pt not in node_index:
            node_index[pt] = len(node_coords)
            node_coords.append(pt)
        point_counts[pt] += 1

    # Step 3: Build patent frequency vector
    variance.patent_count = [point_counts[pt] for pt in node_coords]

    # Step 5: Run clustering algorithm
    start_time = time.time()
    variance.run(node_coords, radii)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.3f} seconds")

    # Step 6: Load results 
    return pd.read_csv(output_file)


# ----------------------------------------
# Write NetworkX DiGraph to Pajek format
# ----------------------------------------
def write_pajek(g, filename):
    """
    Exports a directed graph to Pajek format, including node coordinates.
    """
    node_index = {k: i+1 for i, k in enumerate(g.nodes())}
    with open(filename, 'w+') as f:
        f.write(f'*Vertices {len(node_index)}\n')
        for k in node_index:
            x, y = g.nodes[k].get('pos', (0.0, 0.0))
            f.write(f'{node_index[k]} "{k}" {x:.6f} {y:.6f} 0.0\n')
        f.write('*Arcs\n')
        for i, j in g.edges():
            if j in node_index:
                f.write(f'{node_index[i]} {node_index[j]} 1 1\n')


# ----------------------------------------
# Remove a percentage of edges at random
# ----------------------------------------
def remove_random_edges(G, percentage=10.0):
    """
    Removes a percentage of edges at random from the graph.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    G_copy = G.copy()
    num_edges_to_remove = int(len(G_copy.edges) * (percentage / 100))
    edges_to_remove = random.sample(list(G_copy.edges), num_edges_to_remove)
    G_copy.remove_edges_from(edges_to_remove)
    return G_copy


# ----------------------------------------
# US mainland boundary for context map
# ----------------------------------------
us_outline = [(-125, 24), (-125, 50), (-66, 50), (-66, 24), (-125, 24)]


# ----------------------------------------
# Plot function: Nodes + Edges by Component
# ----------------------------------------
def plot_largest_components_colored(ax, graph, top_k=10, color_map='tab10', 
        linewidth = 0.2, linealpha = 0.8, nodesize = 6 ):
    """
    Plots the graph's nodes and edges, colored by connected components.
    Top_k largest components are assigned distinct bright colors.
    Smaller components are shown in gray.
    """
    # Compute connected components from undirected version
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    components_sorted = sorted(components, key=len, reverse=True)

    # Map each node to its component index
    node_to_component = {}
    for idx, comp in enumerate(components_sorted):
        for node in comp:
            node_to_component[node] = idx

    # Color map
    cmap = plt.get_cmap(color_map)
    default_color = (0.7, 0.7, 0.7)  # gray for small components

    # -------- Plot Edges --------
    edge_segments = []
    edge_colors = []
    for u, v in graph.edges():
        if u in graph.nodes and v in graph.nodes:
            pos_u = graph.nodes[u].get('pos')
            pos_v = graph.nodes[v].get('pos')
            if pos_u and pos_v:
                edge_segments.append([pos_u, pos_v])
                comp_id = node_to_component.get(u, -1)
                color = cmap(comp_id % 10) if comp_id < top_k else default_color
                edge_colors.append(color)
    edge_collection = LineCollection(edge_segments, colors=edge_colors, linewidths=linewidth, alpha=linealpha)
    ax.add_collection(edge_collection)

    # -------- Plot Nodes --------
    xs, ys, node_colors = [], [], []
    for node in graph.nodes():
        pos = graph.nodes[node].get('pos')
        if pos:
            xs.append(pos[0])
            ys.append(pos[1])
            comp_id = node_to_component.get(node, -1)
            color = cmap(comp_id % 10) if comp_id < top_k else default_color
            node_colors.append(color)
    ax.scatter(xs, ys, c=node_colors, s=nodesize)

    # -------- Plot US Outline --------
    if 0:
        us_lon, us_lat = zip(*us_outline)
        ax.plot(us_lon, us_lat, color='black', linewidth=1.0, linestyle='--')

    #ax.set_aspect('equal')
    #ax.grid(True)


# ----------------------------------------
# Main class for generating geometric graphs
# ----------------------------------------


class GeometricNetwork:
    def __init__(self):
        self.minx = +1e6
        self.miny = +1e6
        self.maxx = -1e6
        self.maxy = -1e6
    
    def get_bbox(self, points):
        coords = np.array([(p.x, p.y) for p in points])
        self.minx, self.miny = np.min(coords, axis=0)
        self.maxx, self.maxy = np.max(coords, axis=0)
        print(f"minx={self.minx}\nmaxx={self.maxx}\nminy={self.miny}\nmaxy={self.maxy}")   

    def getFromPointArray (self, points_array,  radius):
        """
        Generate geometric graph(s) over varying distance thresholds.
        """
        points = []
        node_index = {}
        for tup in points_array:
            pt = Point(*tup)
            if pt not in node_index:
                node_index[pt] = len(points)
                points.append(pt)
                
        self.get_bbox(points)
        padding = 1e-6
        boundary = Rectangle(self.minx - padding, self.miny - padding,
                             (self.maxx - self.minx) + 2 * padding,
                             (self.maxy - self.miny) + 2 * padding)
        tree = QuadTreeNode(boundary)
        G = nx.DiGraph()
        for i, pt in enumerate(points):
            tree.insert(pt, i)
            G.add_node(i, pos=(pt.x, pt.y))
        for i, pt in enumerate(points):
                neighbors = tree.query_circle(pt, radius)
                for qti in neighbors:
                    idx = qti.index
                    if i != idx and not G.has_edge(i, idx):
                        G.add_edge(i, idx)
        return G

class ExportGeometricNetworks(ABC):
    def __init__(self, _prefix="./net_"):
        self.minx = +1e6
        self.miny = +1e6
        self.maxx = -1e6
        self.maxy = -1e6
        self.prefix = _prefix

    def get_bbox(self, points):
        coords = np.array([(p.x, p.y) for p in points])
        self.minx, self.miny = np.min(coords, axis=0)
        self.maxx, self.maxy = np.max(coords, axis=0)
        print(f"minx={self.minx}\nmaxx={self.maxx}\nminy={self.miny}\nmaxy={self.maxy}")

    def run(self, points_array, radii, obs=None):
        """
        Generate geometric graph(s) over varying distance thresholds.
        """
        points = []
        node_index = {}
        for tup in points_array:
            pt = Point(*tup)
            if pt not in node_index:
                node_index[pt] = len(points)
                points.append(pt)

        self.init()
        self.get_bbox(points)

        padding = 1e-6
        boundary = Rectangle(self.minx - padding, self.miny - padding,
                             (self.maxx - self.minx) + 2 * padding,
                             (self.maxy - self.miny) + 2 * padding)

        tree = QuadTreeNode(boundary)
        G = nx.DiGraph()
        for i, pt in enumerate(points):
            tree.insert(pt, i)
            G.add_node(i, pos=(pt.x, pt.y))

        for radius in tqdm(radii, desc="Processing radii"):
            edge_count_before = G.number_of_edges()
            for i, pt in enumerate(points):
                neighbors = tree.query_circle(pt, radius)
                for qti in neighbors:
                    idx = qti.index
                    if i != idx and not G.has_edge(i, idx):
                        G.add_edge(i, idx)
            new_edges = G.number_of_edges() - edge_count_before
            self.process_groups(G, radius, obs)
        self.done()
        return G

    def init(self):
        pass

    def done(self):
        pass

    def process_groups(self, G, radius, obs=None):
        print(f"radius={radius:.4f}, nodes={len(G)}, edges={G.number_of_edges()}")
        # Example exports:
        # write_pajek(G, self.prefix + f"{radius:.04f}.net")
        # Gsamp = remove_random_edges(G, 80)
        # write_pajek(Gsamp, self.prefix + "samp_" + f"{radius:.04f}.net")




# ------------ PROJECT SPECIFIC (FIG6) ------------

def load_simulation(p = 3, v = 1, Lambda = 0.0004, df = 1, replicate = 0, 
            data_folder = "/Users/sergi/Dropbox/POLYA_LEVY_CODE/Figure 6/Data/" ):
    path = data_folder + 'PolyaLevy_Entrenchment_p' + str(p) + '_v' + str(v) + '_r' + str(replicate)
    point_array = []
    with open(path+'.txt', 'r') as file:
        first = False
        for line in file:
            if first:
                coords = line[1:-2].split(',')
                city = [-float(coords[0]),float(coords[1])]
                #print(line, coords, city
                point_array.append ( city )
            else:
                first = True
    return point_array
