from __future__ import annotations
import sys
sys.path.append('./ViTSP')

import LLM_TSP.proc_logging
import multiprocessing as mp
import time
from queue import Empty, Full
from typing import Iterable, List, Tuple, Union, Any, Dict
import argparse
import psutil
import os
from LLM_TSP.tsp import TravelingSalesmenProblem, SubTSP
from exact_concorde.exact_concorde import Concorde
from exact_Gurobi.gurobi_model import GurobiTSPModel
from LLM_TSP.proc_logging import _configure_logging
import logging
from dataclasses import dataclass, asdict

__all__ = [
    "evaluate_best_gain",
    "subproblem_scale_below_threshold",
    "non_overlapping",
    "subproblem_overlapping",
    "process_subTSP",
    "solver_master",
    "reformulate_and_solve_subTSP",
]

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
Coordinate = Tuple[int, int, int, int]

@dataclass
class GlobalObjRecord:
    """
    One snapshot of the running objective trajectory produced
    by the solver / LLM pipeline.
    """
    # seconds since the true experiment start
    latency   : float
    # objective value *after* applying the sub‑solution
    new_obj : int
    # rectangular regions that defined the sub‑problem
    coords  : List[Coordinate]
    # |removed_nodes|
    num_nodes_removed : int
    # which LLM (or algo) proposed this sub‑problem
    llm_mode : Any
    # objective value *before* the sub‑problem was solved
    global_solution_version : int
    # which subprocess
    process_name: str = ""

class SubTSPTask:
    def __init__(self, id, current_route, removed_nodes, route_segments, coordinates, parent=False, child=False):
        self.id = id
        self.removed_nodes = removed_nodes  # Ensure uniqueness
        self.num_removed_nodes = len(removed_nodes)
        self.route_segments = route_segments
        self.coordinates = coordinates
        self.new_route = None
        self.new_obj = None
        self.solver_latency = None
        self.gain = None
        self.parent = parent
        self.child = child
        #TODO: the starting and ending index in the current route

def evaluate_best_gain(results, current_obj):
    """
    Given a list of results where each result[1] is the obj value,
    return the best combination with the highest net gain,
    prioritizing lower index if multiple have same gain.
    """
    gain_tasks      = []
    no_impr_tasks   = []

    for task in results:
        gain = max(0, current_obj - task.new_obj)
        task.gain = gain
        print('current gains of %s is %f' % (task.coordinates, gain))

        if gain > 0:
            gain_tasks.append(task)
        else:
            no_impr_tasks.append(task)

    gain_tasks = sorted(gain_tasks, key=lambda task: (-task.gain, task.id)) # task with higher gain and lower id is ranked first

    return gain_tasks, no_impr_tasks

def splice_longest_subroute(par: "Child", sub: "Child") -> None:
    """
    Replace the block in *par.new_route* delimited by the first and last
    occurrence of `sub.removed_nodes` with the corresponding block extracted
    from *sub.new_route*.

    ── Arguments ─────────────────────────────────────────────────────────────
    par  : parent  (childs_with_gain[0])
    sub  : donor   (childs_with_gain[1]); supplies both `removed_nodes`
           and the order to graft in.

    Runs in O(n) time where n = len(route).  Routes are mutated in place.
    Raises ValueError if any removed‑node is missing.
    """
    needle = set(sub.removed_nodes)
    if not needle:
        return                                   # nothing to do

    def first_last(route, items):
        """Return (first_idx, last_idx) where any item of *items* occurs."""
        first = last = None
        for i, v in enumerate(route):
            if v in items:
                if first is None:
                    first = i
                last = i
        if first is None:
            raise ValueError("Some removed‑nodes not found in a route.")
        return first, last

    # --- 1. donor segment ---------------------------------------------------
    d0, d1 = first_last(sub.new_route, needle)
    donor_segment = sub.new_route[d0:d1 + 1]

    # --- 2. parent segment to be replaced -----------------------------------
    p0, p1 = first_last(par.new_route, needle)

    # Optional sanity check: same multiset of vertices
    # if Counter(donor_segment) != Counter(par.new_route[p0:p1 + 1]):
    #     raise ValueError("Segment substitution would change vertex set")

    # --- 3. splice in one go -------------------------------------------------
    par.new_route[p0:p1 + 1] = donor_segment

def exclusive_include(childs_with_gain):
    # TODO: and consecutive sequence
    # all nodes are in the same segment, even though nodes are not consecutive
    for route_segment in childs_with_gain[0].route_segments:
        if set(route_segment) >= set(childs_with_gain[1].removed_nodes):
            pos = {value: idx for idx, value in enumerate(route_segment)}
            try:
                idxs = sorted(pos[x] for x in childs_with_gain[1].removed_nodes)  # KeyError if x not in full
            except KeyError:
                return False
            if len(idxs) != len(set(idxs)):
                return False
            return True
    else:
        return False


def evaluate_gain_contribution(tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent_indicator, child_indicator):

    def has_child_gain_advantage_over_parent(child_tasks, parent_tasks):
        return (
                len(child_tasks) > 1 and
                sum(task.gain for task in child_tasks) > sum(task.gain for task in parent_tasks)
        )

    if len(gain_tasks) <= 1:
        return gain_tasks

    childs_with_gain = []
    parent_with_gain = []

    for task in gain_tasks:
        if task.child == True:
            childs_with_gain.append(task)
        elif task.parent == True:
            parent_with_gain.append(task)

    if has_child_gain_advantage_over_parent(childs_with_gain, parent_with_gain):
        if exclusive_include(childs_with_gain):

            # # Integrate multiple sub-route into the global route
            # needle = set(childs_with_gain[1].removed_nodes)
            # new_subroute = [x for x in childs_with_gain[1].new_route if x in needle]
            #
            # positions = [i for i, v in enumerate(childs_with_gain[0].new_route) if v in needle]  # [1, 2, 3]
            # if len(new_subroute) != len(positions):
            #     raise ValueError("Lists don’t contain exactly the same elements "
            #                      "or they appear multiple times.")
            # for idx, val in zip(positions, new_subroute):
            #     childs_with_gain[0].new_route[idx] = val

            splice_longest_subroute(childs_with_gain[0], childs_with_gain[1])
            childs_with_gain[0].new_obj = tsp_instance.calculate_total_distance(childs_with_gain[0].new_route)
            childs_with_gain[0].coordinates = childs_with_gain[0].coordinates + childs_with_gain[1].coordinates
            childs_with_gain[0].solver_latency = max(childs_with_gain[0].solver_latency, childs_with_gain[1].solver_latency)
            childs_with_gain[0].removed_nodes = childs_with_gain[0].removed_nodes.union(childs_with_gain[1].removed_nodes)
            childs_with_gain[0].num_removed_nodes = len(childs_with_gain[0].removed_nodes)
            childs_with_gain[0].gain =  max(0, current_obj - childs_with_gain[0].new_obj)
            gain_tasks = childs_with_gain

    else:
        gain_tasks = gain_tasks[:1]

    return gain_tasks

def subproblem_scale_below_threshold(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count < args.max_node_for_solver:
        return True
    else:
        return False

def non_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if unique_removed_node_count == len([node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False
def subproblem_overlapping(args, unique_removed_node_count, removed_nodes_list):
    if args.select_sequence:
        return True
    if unique_removed_node_count < len(
        [node for sublist in removed_nodes_list for node in sublist]):
        return True
    else:
        return False

def set_cpu_affinity(process, cpu_ids):
    """
    Set CPU affinity for a process.
    """
    p = psutil.Process(process.pid)
    p.cpu_affinity(cpu_ids)  # Assign specific CPUs to the process

def assign_proportional_cores_to_tasks(tasks: List[Any]) -> Dict[int, List[int]]:
    """
    Proportionally assign available CPU cores to the given list of tasks.
    """
    total_cores = os.cpu_count() or 1
    total_task_lengths = sum(task.num_removed_nodes for task in tasks)

    core_assignment = {}
    current_core = 4

    for i, task in enumerate(tasks):
        weight = task.num_removed_nodes / total_task_lengths if total_task_lengths > 0 else 1 / len(tasks)
        allocated_cores = max(1, int(round(weight * total_cores)))
        core_assignment[i] = list(range(current_core, min(current_core + allocated_cores, total_cores)))
        current_core += allocated_cores
        if current_core >= total_cores:
            break

    return core_assignment

# --------------------------------------------------------------------------- #
# Core worker                                                                 #
# --------------------------------------------------------------------------- #
def reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes):

    sub_tsp = SubTSP(global_distance_mat=tsp_instance.distance_mat,
                     route_segments=route_segments,
                     free_nodes=removed_nodes,
                     starting_point=current_route[0])
    sub_tsp.reformulate_as_ATSP() # build an ATSP distance mat

    start_time = time.time()
    stsp_mat = sub_tsp.transform_partial_ATSP_into_STSP(ATSP_dist_mat=sub_tsp.distance_mat)
    print('Runtime for partial ATSP is: ', time.time() - start_time)

    start_time = time.time()
    # new_route, new_obj, solver_name = parallel_solvers(args, stsp_mat, sub_tsp, tsp_instance)
    if args.solver_model == 'concorde' and len(removed_nodes) >= 5:
        try:
            sub_solver_model = Concorde(dist_matrix=stsp_mat)
            sub_solver_model.optimize(timelimit=args.SolverTimeLimit, verbose=False)
            optimized_sub_tsp = sub_tsp.filter_dummy_nodes(route=sub_solver_model.get_tsp_route(),
                                                           num_ori_nodes=len(sub_tsp.node_list))
            new_route = sub_tsp.resume_master_route(optimized_sub_tsp)
            new_obj = tsp_instance.calculate_total_distance(new_route)
        except Exception as e:
            # Log the error with additional context and fallback gracefully
            print(f"Error occurred during Concorde optimization: {e}")
            print("Falling back to current route and objective.")

            # Fallback to the current route and objective
            new_route = current_route
            new_obj = current_obj

    elif len(removed_nodes) < 5:
        sub_solver_model = GurobiTSPModel(nodes=sub_tsp.node_list,
                                          distance_mat=sub_tsp.distance_mat)
        # sub_solver_model.update_model_param(args)
        sub_solver_model.optimize()
        new_route = sub_tsp.resume_master_route(sub_solver_model.get_tsp_route()[:-1])
        new_obj = tsp_instance.calculate_total_distance(new_route)

    solver_latency = time.time() - start_time

    return new_route, new_obj, solver_latency


def process_subTSP(args, task, tsp_instance, current_route, current_obj, result_queue):
    """
    Simulate a CPU-intensive sub-TSP computation.
    """
    import queue  # Import the queue module

    removed_nodes, route_segments = task.removed_nodes, task.route_segments

    try:
        results = reformulate_and_solve_subTSP(args, tsp_instance, current_route, current_obj, route_segments, removed_nodes)
    except Exception as e:
        print(f"[process_subTSP] Error solving subproblem: {e}")
        task.new_route = current_route
        task.new_obj = current_obj
        task.solver_latency = 0.0
        try:
            result_queue.put(task, block=False)
        except queue.Full:
            pass
        return

    task.new_route      = results[0]
    task.new_obj        = results[1]
    task.solver_latency = results[2]
    try:
        result_queue.put(task, block=False)
    except queue.Full:
        print("Queue is full! Dropping result or retrying...")


def format_task_traj(task):
    coord_str = " ".join(
        f"<coordinates> x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max} </coordinates>"
        for x_min, x_max, y_min, y_max in task.coordinates
    )
    return (
        f"{coord_str}, number of nodes within the subrectangle={task.num_removed_nodes}, "
        f"travel distance reduction={round(task.gain, 2)} (the higher the better), "
        f"computation time for this subrectangle={round(task.solver_latency, 2)} sec \n"
    )

def solver_master(args, initialization_latency, tsp_instance, pending_subproblem_queue, pending_re_subproblem_queue, global_obj, global_sol, obj_lock, sol_lock, traj_lock, selection_traj, deadline, optimization_traj_queue, t0):
    _configure_logging()
    log = logging.getLogger()
    optimization_traj = []

    optimization_traj.append((initialization_latency, global_obj.value, None, None, None))

    t0 = time.time()
    deadline = t0 + args.total_time_budget
    llm_source = 'reasoning'
    while time.time() < deadline:
        remaining = max(deadline - time.time(), 0.0)
        try:
            with traj_lock:
                subproblem = pending_re_subproblem_queue.get_nowait()
                llm_source = 'reasoning'
        except Empty:
            try:
                with traj_lock:
                    subproblem = pending_subproblem_queue.get(timeout=min(0.01, remaining))
                    llm_source = 'fast_thinking'
            except:
                continue

        if subproblem is None:
            log.info("sentinel received – master exiting")
            return

        removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
        unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, global_obj.value)

        # if overlapping --> must merge, solve as one single TSP problem
        # if non-overlapping, yet smaller than a given threshold --> merge plus separate
        # if non-overlapping, yet greater than a given threshold --> separate
        start_time = time.time()
        with sol_lock:
            current_route = list(global_sol)
        with obj_lock:
            current_obj = global_obj.value

        if subproblem_overlapping(args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
        else:
            if subproblem_scale_below_threshold(args, len(unique_removed_node), removed_nodes_list):
                tasks = []
                for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                    task = SubTSPTask(id=idx,
                                      current_route=current_route,
                                      removed_nodes=removed_nodes,
                                      route_segments=route_segments,
                                      coordinates=[coordinate],
                                      child=True)
                    tasks.append(task)

                merged_removed_nodes = unique_removed_node
                merged_route_segments = tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
                merged_id = tasks[-1].id + 1

                merged_task = SubTSPTask(id=merged_id,
                                         current_route=current_route,
                                         removed_nodes=merged_removed_nodes,
                                         route_segments=merged_route_segments,
                                         coordinates=coordinates_list,
                                         parent=True)

                tasks.append(merged_task)
                parent = True
                child = True
            else:
                # tasks = list(zip(removed_nodes_list, route_segments_list))
                tasks = []
                for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                        zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                    task = SubTSPTask(id=idx,
                                      current_route=current_route,
                                      removed_nodes=removed_nodes,
                                      route_segments=route_segments,
                                      coordinates=[coordinate],
                                      child=True)
                    tasks.append(task)

                parent = False
                child = True

        log.info('Time spent in handle parent and child: %f', time.time() - start_time)

        core_map = assign_proportional_cores_to_tasks(tasks)
        processes = []
        result_queue = mp.Queue()

        start_time = time.time()

        # Spawn multiple processes
        for i, task in enumerate(tasks):
            current_route, current_obj = list(global_sol), global_obj.value
            process = mp.Process(target=process_subTSP, args=(args, task, tsp_instance, current_route, current_obj, result_queue))
            process.start()
            processes.append(process)

            #TODO: Set CPU affinity based on the given CPU cores for concorde master
            set_cpu_affinity(process, core_map[i])

        results = []
        completed_processes = 0
        num_processes = len(tasks)
        wait_deadline = time.time() + args.SolverTimeLimit + 20

        while completed_processes < num_processes:
            if time.time() > wait_deadline:
                log.warning('Subproblem wait timeout, collected %d/%d', completed_processes, num_processes)
                break
            if not result_queue.empty():
                task = result_queue.get()
                results.append(task)
                completed_processes += 1
            else:
                time.sleep(0.05)

        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

        if results:
            # TODO: wrap this chunk to develop an algorithm to accept solutions from multiple subproblems
            # best_gain, best_selection_id, all_gains = evaluate_best_gain(results, current_obj, parent, child)
            gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
            if gain_tasks:
                gain_tasks = evaluate_gain_contribution(tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)
            # new_route, new_obj, solver_latency, _, _ = results[best_selection_id]

            # always use the first task (either merged or as is)
            if gain_tasks:
                new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
                coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
            else:
                new_route, new_obj, solver_latency = current_route, current_obj, 0
                coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = None, None, None

        # ---------------------------------
        # hill-climbing acceptance criteria
        # ---------------------------------
        delta_obj = new_obj - global_obj.value

        if delta_obj < 0:
            with obj_lock:
                old_obj = global_obj.value
                global_obj.value = new_obj
            with sol_lock:
                global_sol[:] = new_route[:]
                optimization_traj.append((round(time.time() - t0 + initialization_latency, 2), new_obj, coordinates_list, len(removed_nodes), llm_source, subproblem.current_obj))
                current_route_edges = tsp_instance.get_route_pair(new_route)
        else:
            with obj_lock:
                old_obj = global_obj.value
                optimization_traj.append((round(time.time() - t0 + initialization_latency, 2), old_obj, coordinates_list, len(removed_nodes), llm_source, subproblem.current_obj))

        if args.keep_selection_trajectory:
            with traj_lock:
                if gain_tasks:
                    selection_traj.put(format_task_traj(gain_tasks[0]))
                for task in no_impr_tasks:
                    selection_traj.put(format_task_traj(task))


        log.info("updated obj %s→%s, using %s", old_obj, global_obj.value, llm_source)

        # ------------------------------------------------------------
        # BEFORE exit put the list on the result queue
        # ------------------------------------------------------------
    try:
        import pandas as pd
        from pathlib import Path

        df = pd.DataFrame(optimization_traj,
                          columns=["Latency", "Objective Values", "Subrectangles", "Num Removed Nodes", "LLM mode",
                                   "Global Solution Version"])
        instance_name = Path(args.instance_path).stem
        df.to_csv(
            f'/local/scratch/a/yin195/vllm-carbon-monitoring/LLM-TSP-async/experiments/LLM_TSP_exp/{instance_name}_max_nodes_{args.max_node_for_solver}_time_budget_{args.total_time_budget}_initial_{args.initial_solution_model}_llm_{args.fast_llm_model}_{args.reasoning_llm_model}_solver_{args.solver_model}_subproblem_{args.llm_subproblem_selection}_include_null_exp.csv',
            index=False)
        # return obj_traj, subproblems
        print('starting to put optimization traj')
        optimization_traj_queue.put_nowait(optimization_traj)
        print('successfully put optimization traj')
    except (BrokenPipeError, EOFError):
        log.warning("could not return obj_traj – queue closed")



def sample_independent_subproblem(config, active_subproblems, gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [gain_subproblem_queue, subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            try:
                subproblem = queue.get_nowait()
            except Empty:
                break # does this put the subproblem back to the queue?
            # TODO: a subproblem can include multiple removed_nodes 
            if all(subproblems_are_independent(active_sp, subproblem, config, current_route) for active_sp in active_subproblems):
                selected_subproblem = subproblem
                print('Retrieved valid subproblems')
                break
            else:
                temp_buffer.append(subproblem)


        for sp in temp_buffer:
            queue.put(sp) # since we selectively choose a subproblem, the order in the queue is not important
        
        if selected_subproblem:
            return selected_subproblem
        
    return None

def sample_next_subproblem(config, active_subproblems, subproblem_re_queue, subproblem_ft_queue, traj_lock, current_route):

    queues_with_priority = [subproblem_re_queue, subproblem_ft_queue]
    for queue in queues_with_priority:
        temp_buffer = []
        selected_subproblem = None

        while True:
            try:
                subproblem = queue.get_nowait()
            except Empty:
                break # does this put the subproblem back to the queue?
            if subproblem:
                return subproblem

    return None


def _is_within_segment(removed_nodes, route_segment):
    # determine if a given (non-consecutive) node list is within a segment
    # i.e., the longest consecutive node sequence speficied by the starting and ending nodes should be within the segment   
    if set(route_segment) >= set(removed_nodes):
        pos = {value: idx for idx, value in enumerate(route_segment)}
        try:
            idxs = sorted(pos[x] for x in removed_nodes)  # KeyError if x not in full
        except KeyError:
            return False
        if len(idxs) != len(set(idxs)):
            return False
        return True

def subproblems_are_independent(running_subproblem, pending_subproblem, config, current_route):
   
    # merge all removed nodes of the running subproblem first 
    unique_removed_node = sorted(set(node for sublist in running_subproblem.removed_nodes_list for node in sublist))
    route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, unique_removed_node)

    # then obtain the unified segments
    # if all removed_nodes is within any segment return true, else, return false
    for removed_nodes in pending_subproblem.removed_nodes_list:
        if not any(_is_within_segment(removed_nodes, route_segment) for route_segment in route_segments):
            return False
    return True
        

def subproblem_solver(subproblem, config):
    _configure_logging()
    log = logging.getLogger()

    removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)
    wait_deadline = time.time() + config.args.SolverTimeLimit + 20

    while completed_processes < num_processes:
        if time.time() > wait_deadline:
            log.warning('Subproblem wait timeout, collected %d/%d', completed_processes, num_processes)
            break
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1
        else:
            time.sleep(0.05)

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        # if subproblem.solution_version != config.global_obj.value:
        #     log.info("Find different version of solution")
        # elif subproblem.solution_version == config.global_obj.value:
        delta_obj = new_obj - config.global_obj.value

        if delta_obj < 0:
            with config.obj_lock:
                old_obj = config.global_obj.value
                config.global_obj.value = new_obj
            with config.sol_lock:
                config.global_sol[:] = new_route[:]
        else:
            with config.obj_lock:
                old_obj = config.global_obj.value

        with config.obj_lock:
            now = round(time.time() - config.t0 + config.warmstart_latency, 2)
            proc = mp.current_process()
            record = GlobalObjRecord(
                                    latency=now,
                                    new_obj=config.global_obj.value,
                                    coords=coordinates_list,
                                    num_nodes_removed=len(removed_nodes),
                                    llm_mode=subproblem.llm_source,
                                    global_solution_version=subproblem.current_obj,
                                    process_name=proc.name,
                                    )

        config.track_global_obj_queue.put(record)
        
        if config.args.keep_selection_trajectory:
            with config.traj_lock:
                if gain_tasks:
                    config.selection_traj.put(format_task_traj(gain_tasks[0]))
                for task in no_impr_tasks:
                    config.selection_traj.put(format_task_traj(task))

    with config.obj_lock:
        log.info("updated obj %s→%s, using %s", old_obj, config.global_obj.value, subproblem.llm_source)


def subproblem_verifier(subproblem, config):
    _configure_logging()
    log = logging.getLogger()

    removed_nodes_list, route_segments_list, coordinates_list = subproblem.removed_nodes_list, subproblem.route_segments_list, subproblem.coordinates_list
    unique_removed_node = sorted(set(node for sublist in removed_nodes_list for node in sublist))

    with config.obj_lock:
        current_obj = config.global_obj.value
        log.info("start coords=%s obj_in=%s", subproblem.coordinates_list, current_obj)
    with config.sol_lock:
        current_route = list(config.global_sol)

    # ---- determine how to handle multiple selections    
    if subproblem_overlapping(config.args, len(unique_removed_node), removed_nodes_list):
            removed_nodes = unique_removed_node  # multiple subproblems get merged
            route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, removed_nodes)
            task = SubTSPTask(id=0,
                              current_route=current_route,
                              removed_nodes=removed_nodes,
                              route_segments=route_segments,
                              coordinates=coordinates_list,
                              parent=True)
            tasks = [task]
            parent = True
            child = False
    else:
        if subproblem_scale_below_threshold(config.args, len(unique_removed_node), removed_nodes_list):
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            merged_removed_nodes = unique_removed_node
            merged_route_segments = config.tsp_instance.remove_edges_given_nodes(current_route, merged_removed_nodes)
            merged_id = tasks[-1].id + 1

            merged_task = SubTSPTask(id=merged_id,
                                        current_route=current_route,
                                        removed_nodes=merged_removed_nodes,
                                        route_segments=merged_route_segments,
                                        coordinates=coordinates_list,
                                        parent=True)

            tasks.append(merged_task)
            parent = True
            child = True
        else:
            # tasks = list(zip(removed_nodes_list, route_segments_list))
            tasks = []
            for idx, (removed_nodes, route_segments, coordinate) in enumerate(
                    zip(removed_nodes_list, route_segments_list, coordinates_list), start=0):
                task = SubTSPTask(id=idx,
                                    current_route=current_route,
                                    removed_nodes=removed_nodes,
                                    route_segments=route_segments,
                                    coordinates=[coordinate],
                                    child=True)
                tasks.append(task)

            parent = False
            child = True

    core_map = assign_proportional_cores_to_tasks(tasks)
    processes = []
    result_queue = mp.Queue()

    start_time = time.time()

    # Spawn multiple processes
    for i, task in enumerate(tasks):
        # current_route, current_obj = list(global_sol), global_obj.value
        process = mp.Process(target=process_subTSP, args=(config.args, task, config.tsp_instance, current_route, current_obj, result_queue))
        process.start()
        processes.append(process)

        #TODO: Set CPU affinity based on the given CPU cores for concorde master
        set_cpu_affinity(process, core_map[i])

    results = []
    completed_processes = 0
    num_processes = len(tasks)
    wait_deadline = time.time() + config.args.SolverTimeLimit + 20

    while completed_processes < num_processes:
        if time.time() > wait_deadline:
            log.warning('Subproblem wait timeout, collected %d/%d', completed_processes, num_processes)
            break
        if not result_queue.empty():
            task = result_queue.get()
            results.append(task)
            completed_processes += 1
        else:
            time.sleep(0.05)

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    log.info('Time spent in multiprocessing subTSPs: %f', time.time() - start_time)

    if results:
        gain_tasks, no_impr_tasks = evaluate_best_gain(results, current_obj)
        if gain_tasks:
            gain_tasks = evaluate_gain_contribution(config.tsp_instance, current_obj, current_route, gain_tasks, no_impr_tasks, parent, child)

        # always use the first task (either merged or as is)
        if gain_tasks:
            new_route, new_obj, solver_latency = gain_tasks[0].new_route, gain_tasks[0].new_obj, gain_tasks[0].solver_latency
            coordinates_list, removed_nodes, route_segments = gain_tasks[0].coordinates, gain_tasks[0].removed_nodes, gain_tasks[0].route_segments
        else:
            new_route, new_obj, solver_latency = current_route, current_obj, 0
            coordinates_list, removed_nodes, route_segments = subproblem.coordinates_list, [node for sublist in subproblem.removed_nodes_list for node in sublist], subproblem.route_segments_list
    else:
        new_route, new_obj, solver_latency = current_route, current_obj, 0
        coordinates_list, removed_nodes, route_segments = None, None, None

    # ---------------------------------
    # hill-climbing acceptance criteria
    # ---------------------------------
    with config.obj_lock, config.sol_lock:
        
        delta_obj = new_obj - config.global_obj.value

        if delta_obj < 0:
            config.gain_subproblem_queue.put(subproblem)
            print('Finding a gain subproblem!')