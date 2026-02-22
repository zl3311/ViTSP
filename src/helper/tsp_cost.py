#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 10/18/24 5:30 PM
@File:tsp_cost.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''
import math
from codes.helper.parse_llm_response import clean_trace_content_from_llm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calculate_total_distance(coordinates, tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        point1 = coordinates[tour[i]]
        point2 = coordinates[tour[i + 1]]
        total_distance += euclidean_distance(point1, point2)

    # Add distance from the last point back to the first point to complete the tour
    total_distance += euclidean_distance(coordinates[tour[-1]], coordinates[tour[0]])

    return total_distance

def compute_trace_distance(trace):
    if trace and  len(trace) == len(coordinates):
            return calculate_total_distance(coordinates, trace)
    return None

def calculate_distance_dict(nodes):
    distance_dict = {}
    for i in nodes:
        for j in nodes:
            if i != j:  # Avoid calculating distance from a node to itself
                # Calculate Euclidean distance
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # Store distance in a nested dictionary
                distance_dict[(i, j)] = distance
            else:
                distance_dict[(i, j)] = 0
    return distance_dict


def calculate_distance_matrix(nodes):
    # Initialize distance matrix
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    M = 1e6

    # Calculate Euclidean distance between each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i, j] = M  # Mark the diagonal as 'M'
            else:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                distance_matrix[i, j] = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance_matrix

def plot_nodes(coordinates, tour):
    # Tour sequence
    # tour = [0, 16, 11, 14, 15, 13, 7, 9, 1, 8, 3, 10, 6, 19, 12, 4, 5, 2, 17, 18, 0]
    # Plot the nodes
    plt.figure(figsize=(20, 20))
    for idx, (x, y) in enumerate(coordinates):
        # Mark the starting point with a red star
        if idx == 0:
            plt.scatter(x, y, color='red', marker='*', s=150, zorder=5, label="Starting Point (Node 0)")
        else:
            plt.scatter(x, y, color='#00668c', marker='o', zorder=5)

        # Annotate each node with its index
        # plt.text(x, y, f'{idx}', fontsize=15, ha='right', color='black')

    # Draw arrows between nodes following the tour
    for i in range(len(tour) - 1):
        start_idx, end_idx = tour[i], tour[i + 1]
        start_x, start_y = coordinates[start_idx]
        end_x, end_y = coordinates[end_idx]

        plt.annotate(
            '', xy=(end_x, end_y), xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5)
        )

    # Set x and y ticks to create a detailed grid with increments of 5
    plt.xticks(range(0, 1001, 100), rotation=45)
    plt.yticks(range(0, 1001, 100), rotation=0)

    # Add labels, legend, and grid for better visualization
    plt.title("Traveling Salesman Problem Tour")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
