#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 11/11/24 10:49 PM
@File:generate_instance.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''
import numpy as np
import argparse
from pathlib import Path
import time
import random


def generate_coordinates(args):
    """
    Generate node coordinates for a TSP instance.
    Supports both uniform and clustered distributions.
    """
    if args.n_clusters > 0:
        coordinates = clustered_coordinates(args)
    else:
        coordinates = np.random.uniform(0, args.max_xy, size=(args.n_nodes, 2))
    return coordinates


def clustered_coordinates(args, max_xy=1):
    """
    Generates clustered coordinates with optional random uniform points.
    """
    # Fraction of points that are uniformly distributed (if mixed)
    uniform_frac = 0.5 if args.mixed else 0.0
    n_uniform = int(args.n_nodes * uniform_frac)
    n_clustered = args.n_nodes - n_uniform

    # Uniformly distributed points (if any)
    uniform_locs = np.random.uniform(0, max_xy, size=(n_uniform, 2))

    # Clustered points generation
    centers = np.random.uniform(0.2, max_xy - 0.2, size=(args.n_clusters, 2))
    clustered_locs = []
    for _ in range(n_clustered):
        center = centers[np.random.randint(0, len(centers))]
        point = np.random.normal(center, args.std_cluster)
        while not ((0 <= point[0] <= max_xy) and (0 <= point[1] <= max_xy)):
            point = np.random.normal(center, args.std_cluster)  # Regenerate if out of bounds
        clustered_locs.append(point)

    # Combine uniform and clustered points
    all_coordinates = np.vstack((uniform_locs, clustered_locs))
    return all_coordinates


def generate_tsp_instance(args, instance_id):
    """
    Generates a TSP instance and saves it in a format compatible with TSP solvers.
    """
    if args.n_clusters >= 1:
        # Randomly set n_clusters and std_cluster for each instance
        args.n_clusters = random.randint(1, 5)
        args.std_cluster = random.uniform(0.05, 0.1)

    dist_type = (
        "uniform" if args.n_clusters == 0
        else "mixed" if args.mixed
        else "clustered"
    )
    folder_name = f"{dist_type}_n{args.n_nodes}"
    folder_path = Path(args.output_path) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    output_file = folder_path / f'instance_{instance_id}.tsp'
    coordinates = generate_coordinates(args)

    with open(output_file, 'w') as file:
        file.write(f"NAME : {args.name}_{dist_type}_{instance_id}\n")
        file.write("TYPE : TSP\n")
        file.write(f"DIMENSION : {args.n_nodes}\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coordinates, start=1):
            file.write(f"{i} {x:.6f} {y:.6f}\n")
        file.write("EOF\n")
    print(f"TSP instance {args.name} with {args.n_nodes} nodes generated at {args.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=Path, help='Path to save the generated TSP instance')
    parser.add_argument('--name', type=str, default='tsp_instance', help='Name of the TSP instance')
    parser.add_argument('--n_nodes', type=int, default=10000, help='Number of nodes')
    parser.add_argument('--n_clusters', type=int, default=0, help='Number of clusters')
    parser.add_argument('--mixed', action='store_true', help='Mix clustered and uniform distributions')
    parser.add_argument('--std_cluster', type=float, default=0.07, help='Standard deviation for clustering')
    parser.add_argument('--max_xy', type=float, default=1.0, help='Maximum coordinate value for x and y')
    parser.add_argument('--n_instances', type=int, default=100, help='Number of instances to generate')

    args = parser.parse_args()

    for instance_id in range(args.n_instances):
        generate_tsp_instance(args, instance_id)

    print(f'Generating {args.n_instances} TSP instances from '
          f'{"uniform distribution" if args.n_clusters == 0 else f"mixed distribution with {args.n_clusters} clusters" if args.mixed else f"clustered distribution with {args.n_clusters} clusters"}, '
          f'for each instance containing {args.n_nodes} nodes and saved to {args.output_path}', flush=True)


if __name__ == '__main__':
    main()
