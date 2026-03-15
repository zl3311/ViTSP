# ViTSP: A Vision Language Models Guided Framework for Solving Large-Scale Traveling Salesman Problems

<div align="center">
  
[![ICLR-Brazil](https://img.shields.io/badge/ICLR-Brazil-green)](https://iclr.cc/Conferences/2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/ICLR-Poster-blue)](#paper)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-orange)](https://openreview.net/forum?id=2LoaiaGKuV&noteId=Amm8FnWJZa)

</div>

This is the repo for ViTSP accepted by **ICLR 2026**.

## 🚀 TL;DR

Learning-based methods have shown promise in combinatorial optimization, but they often fail when problem instances differ in scale or structure from those seen during training. ViTSP is a **hybrid GenAI–OR framework**, transforming routing instances into visual representations and allowing Vision Language Models (VLMs) to propose meaningful subproblem decompositions. These subproblems are asynchronously solved by exact OR solvers, yielding high-quality solutions in practice without task-specific training.

<p align="center">
  <img src="./ViTSP_framework.png" width="700">
</p>

---

## ✨ Highlights

- Vision-language guided decomposition heuristics for large routing instances

- Exact OR solvers for local optimality guarantees

- Asynchronous system design for heterogeneous compute pipelines

- Training-free generalization across TSP scales, outperforming learning-based methods

---
## Citation
If you find our work useful, please cite:
```
@inproceedings{
yin2026vitsp,
title={Vi{TSP}: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems},
author={Zhuoli Yin and Yi Ding and Reem Khir and Hua Cai},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2LoaiaGKuV}
}
```
---

## Repository Structure
```
ViTSP_ICLR2026/
├── README.md
├── requirements.txt            # Python dependencies — pip install -r requirements.txt
├── data/
│   └── instances/
│       ├── tsplib_original/        # 144 TSPLIB benchmark instances (.tsp + .opt.tour)
│       └── my_tsps/                # 16 custom large instances (BIGTSP_10000_*.tsp)
│
└── src/                            # All source code — run from this directory
    ├── LLM_TSP/                    # Core ViTSP pipeline
    │   ├── main.py                 # ★ Main entry point
    │   ├── config.py               # LLMConfig & SolverConfig dataclasses
    │   ├── ablation_config.py      # Per-instance max_nodes & time budgets
    │   ├── llm.py                  # OpenAI GPT wrapper (vision_chat)
    │   ├── tsp.py                  # TSP data structure, distance computation, SubTSP reformulation
    │   ├── initial_solution.py     # LKH / Farthest-Insertion warm-start
    │   ├── selector.py             # RandomSelector fallback
    │   ├── clean_temp_files.py     # Post-run cleanup
    │   ├── proc_logging.py         # Logging config
    │   ├── llm_selector/
    │   │   └── llm_selector.py     # Async LLM producer, visual selection, subproblem handling
    │   └── solver/
    │       └── solver.py           # Concorde/Gurobi subproblem solver, gain evaluation, hill climbing
    │
    ├── exact_concorde/
    │   └── exact_concorde.py       # Concorde TSP solver wrapper (PyConcorde)
    │
    ├── exact_Gurobi/
    │   └── gurobi_model.py         # Gurobi MIP model for TSP (with lazy subtour elimination)
    │
    ├── heuristic_LKH/
    │   ├── heuristic_LKH.py        # LKH-3 solver wrapper
    │   └── LKH_param_sweeping_in_parallel.py  # LKH hyperparameter sweep utility
    │
    └── helper/
        ├── parse_instances.py      # TSPLIB / CVRP file parser
        ├── parse_llm_response.py   # Extracts <coordinates> from LLM responses
        ├── plot_solution.py        # Matplotlib & Plotly TSP visualization
        ├── generate_instance.py    # Random / clustered TSP instance generator
        ├── tsp_cost.py             # Cost utilities
        ├── tsp_generate_instance.py
        └── tracker.py
```
---

## Prerequisites

### 1. Python Environment

This project requires a Linux or Windows environment. The LKH solver does not currently support MacOS.

```bash
# Clone the repository
git clone https://github.itap.purdue.edu/uSMART/ViTSP_ICLR2026.git
cd ViTSP_ICLR2026

# Create conda environment (Python 3.10+ recommended)
conda create -n vitsp python=3.10 -y
conda activate vitsp

# Install all Python dependencies
pip install -r requirements.txt
```

### 2. LKH-3 Solver (Initial Solution)

LKH-3 is used to generate high-quality initial TSP tours. ViTSP expects the binary at `./LKH-3.0.13/LKH` relative to the project root.


Their website is available at: http://webhotel4.ruc.dk/~keld/research/LKH-3/
```bash
# Download and build LKH-3
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz
tar xzf LKH-3.0.13.tgz
cd LKH-3.0.13
make
cd ..

# Verify the executable — the code expects this exact path:
ls ./LKH-3.0.13/LKH    # should show the binary
```

> **Note:** The solver path is hardcoded in `src/LLM_TSP/initial_solution.py` (line 33) as `'./LKH-3.0.13/LKH'`. If you install LKH elsewhere, update this path accordingly.

### 3. Concorde Solver (Subproblem Re-optimization)

Concorde, the SOTA exact TSP solver, is the default solver for re-optimizing subproblems selected by the VLM. Install via PyConcorde:

```bash
pip install pyconcorde
```

To build Concorde from source (alternative):

```bash
# Requires QSopt LP solver: http://www.math.uwaterloo.ca/~bico/qsopt/
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar xzf co031219.tgz
cd concorde
./configure --with-qsopt=/path/to/qsopt
make
```

Verify installation:

```bash
python -c "from concorde.tsp import TSPSolver; print('Concorde OK')"
```

### 4. Gurobi (Optional Safeguard Exact Solver)

Gurobi can be used as a safeguard subproblem solver (set `--solver_model gurobi`). It requires a [license](https://www.gurobi.com/academia/academic-program-and-licenses/).

```bash
# Install
pip install gurobipy
# — OR via conda —
conda install -c gurobi gurobi

# Obtain a free academic license:
# https://www.gurobi.com/academia/academic-program-and-licenses/
grbgetkey YOUR_LICENSE_KEY

# Verify
python -c "import gurobipy; print('Gurobi', gurobipy.gurobi.version())"
```

### 5. OpenAI API Key

ViTSP uses the **OpenAI API** for VLM calls. The framework runs two concurrent LLM processes — a *fast-thinking* model and a *reasoning* model — each requiring an API key.

Open `src/LLM_TSP/main.py` and set your API keys (lines 538–539):

```python
OPENAI_API_1 = 'sk-...'   # API key for fast-thinking LLM
OPENAI_API_2 = 'sk-...'   # API key for reasoning LLM
```

> **Tip:** You can use the same key for both, or use two separate keys to avoid rate-limit issues.

**Supported models** (configured via CLI):

| Role | Default Model | CLI Flag |
|------|--------------|----------|
| Fast-thinking LLM | `gpt-4.1-2025-04-14` | `--fast_llm_model` |
| Reasoning LLM | `o4-mini-2025-04-16` | `--reasoning_llm_model` |

💰 API Cost Reference: In our experiments, the average OpenAI API cost per instance ranged from $0.12 (dsj1000, 1K nodes) to $39.40 (pla85900, 85.9K nodes).

Other tested models include `gpt-4o`, `gpt-4.1-mini-2025-04-14`, and `gpt-5.1-2025-11-13`.

---

## Quick Start

All commands should be run from the `src/` directory:

```bash
cd ViTSP_ICLR2026/src
```

**Run ViTSP on all TSPLIB instances:**

```bash
python -m LLM_TSP.main \
    --instance_path ./data/instances/tsplib_original/ \
    --total_time_budget 100 \
    --fast_llm_model gpt-4.1-2025-04-14 \
    --reasoning_llm_model o4-mini-2025-04-16 \
    --solver_model concorde \
    --initial_solution_model LKH \
    --max_workers 8 \
    --keep_selection_trajectory \
    --llm_subproblem_selection 2
```
Per-instance time budgets can also be customized in `ablation_config.py`. Alternatively, individual instances can be terminated early using a K-consecutive-non-improvement stopping criterion.

**Output:** CSV files saved to `./experiments/ViTSP/` containing the objective trajectory and system profile.

---

### Random Baseline (No VLM as the visual selector)

To run the random sub-rectangle or sub-sequence selection baseline (no API calls needed):

```bash
python -m LLM_TSP.main \
    --fast_llm_model random \
    --reasoning_llm_model random \
    --random_selection \
    --solver_model concorde \
    --initial_solution_model LKH \
    --max_workers 8 \
    --select_sequence \ # optinal for sub-sequence selection
```

### Custom Instances (10K nodes)

The repository includes 16 custom 10,000-node instances:

```bash
python -m LLM_TSP.main \
    --instance_path ./data/instances/my_tsps/BIGTSP_10000_1.tsp \
    --total_time_budget 2000
```

To generate new random instances:

```bash
python -m helper.tsp_generate_instance \
    --output_dir ./data/instances/my_tsps/
```

---

## CLI Reference

All arguments are defined in `src/LLM_TSP/main.py`:

### General

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--instance_path` | str | `./data/instances/tsplib_original` | Path to `.tsp` file or directory |
| `--exp_output_path` | str | `./experiments/ViTSP` | Output directory for experiment CSVs |
| `--total_time_budget` | float | `1000` | Wall-clock time budget in seconds |
| `--max_workers` | int | `8` | Max parallel Concorde solver processes. Can be adjusted based on available CPU cores|

### Initial Solution

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--initial_solution_model` | str | `LKH` | Warm-start method: `LKH` or `FI` (Farthest Insertion) |

### Solver

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--solver_model` | str | `concorde` | Subproblem solver: `concorde` or `gurobi` |
| `--SolverTimeLimit` | float | `10` | Time limit per solver call (seconds) |
| `--max_node_for_solver` | int | `2000` | Max nodes per subproblem (auto-set: 1000 if n < 10K, else 2000) |

### LLM / Selector

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fast_llm_model` | str | `gpt-4.1-2025-04-14` | Fast-thinking VLM model name. Use `random` for no-LLM baseline |
| `--reasoning_llm_model` | str | `o4-mini-2025-04-16` | Reasoning VLM model name. Use `random` for no-LLM baseline |
| `--llm_subproblem_selection` | int | `2` | Number of sub-rectangles the LLM proposes per call |
| `--gridding_resolution` | int | `5` | Grid divisions for reference |
| `--keep_selection_trajectory` | flag | off | Include selection history in LLM prompts |
| `--select_sequence` | flag | off | Select route sequences instead of rectangles |
| `--random_selection` | flag | off | Use random selection (no LLM) |
| `--hard_coded_subrectangle` | flag | off | Use hard-coded sub-rectangles (testing only) |



