# File: experiments/lkh_param_sweep.py

import sys
import time
import os
import csv
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

sys.path.append('/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async')

from helper.parse_instances import FileParser
from LLM_TSP.tsp import TravelingSalesmenProblem
from LLM_TSP.initial_solution import Initializer

def solve_one_param(args_tuple):
    file_name, nodes, distance_mat, problem_path, max_trials, runs, solution_model = args_tuple

    solution_initializer = Initializer(solution_model)
    tsp_instance = TravelingSalesmenProblem(node_coords_dict=nodes, distance_mat=distance_mat)

    start_time = time.time()
    current_route, current_obj = solution_initializer.LKH(
        tsp_instance,
        problem_path,
        max_trials=max_trials,
        runs=runs,
        float_result=False
    )
    latency = time.time() - start_time

    return (file_name, len(nodes), max_trials, runs, latency, current_obj)

def main(args):
    files = os.listdir(args.instance_path)
    file_parser = FileParser()

    for file in tqdm(files):
        if file.startswith('.') or not file.endswith('.tsp'):
            continue

        # dim = file_parser.get_dim_from_filename(file)
        # if dim is None or dim < args.min_nodes or dim > args.max_nodes:
        #     continue

        print(f"Processing {file}")

        instance_info = file_parser.parse_instance_from_file(os.path.join(args.instance_path, file))
        coordinates = instance_info['COORDINATES']
        distance_mat = instance_info['COST_MATRIX']

        if not coordinates or len(coordinates) < args.min_nodes:
            continue

        nodes = {i: (x, y) for i, (x, y) in enumerate(coordinates)} 

        n_nodes = len(nodes)
        trial_increments = list(range(n_nodes, n_nodes + 1, n_nodes))
        run_increments = list(range(args.min_runs, args.max_runs + 1, args.run_step))

        param_combinations = []
        for max_trials in trial_increments:
            for runs in run_increments:
                param_combinations.append((file.split('.')[0], nodes, distance_mat, os.path.join(args.instance_path, file), max_trials, runs, args.solution_model))

        result_file_path = (
            f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LKH/'
            f'{file.split(".")[0]}_param_sweep_{args.solution_model}_time_limit_{args.time_limit_per_run}_max_trails_{args.max_trials}_max_runs_{args.max_runs}.csv'
        )

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(solve_one_param, param_combinations), total=len(param_combinations)))

        with open(result_file_path, mode='w', newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(['Instance', 'Nodes', 'Max_Trials', 'Max_Runs', 'Latency', 'Objective_Value'])
            for result in results:
                writer.writerow(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traveling Salesman Problem LKH Parameter Sweep")
    parser.add_argument('--instance_path', type=str, default='/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/instances/my_tsps', help='Path to the instance files')
    parser.add_argument('--max_nodes', type=int, default=10_001, help='Maximum nodes to solve')
    parser.add_argument('--min_nodes', type=int, default=10_000, help='Minimum nodes to solve')
    parser.add_argument('--solution_model', type=str, default='LKH', help='Solution model to use')
    parser.add_argument('--time_limit_per_run', type=float, default=1_000_000, help='Time limit per run')
    parser.add_argument('--max_trials', type=int, default=10_000, help='Maximum number of trials for LKH sweep')
    parser.add_argument('--trial_step', type=int, default=100, help='Step size for increasing max_trials')
    parser.add_argument('--max_runs', type=int, default=50, help='Maximum value for max_runs sweep')
    parser.add_argument('--min_runs', type=int, default=10, help='Minimum run sweep, by default 10')
    parser.add_argument('--run_step', type=int, default=10, help='Step size for increasing max_runs')

    args = parser.parse_args()
    main(args)
