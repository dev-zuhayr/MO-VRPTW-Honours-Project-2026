MOVRPTW Benchmark Framework
===========================

A modular Python framework for benchmarking metaheuristics (plus an exact
Gurobi baseline) on Solomon's VRPTW instances, with a focus on
multi-objective optimisation. Solvers approximate a Pareto front, the front
is scored with hypervolume and compared against best-known solutions, and
everything is logged to an Excel workbook.

    parse Solomon instance -> run solvers (x seeds x params)
        -> aggregate -> results.xlsx + plots


FEATURES
--------
- Solomon parser (C1/C2/R1/R2/RC1/RC2, any size).
- Six objectives, all minimised: num_vehicles, total_distance, total_wait,
  total_tardiness, route_imbalance, makespan.
- Hard or soft time windows through a single tw_mode setting.
- Eight solvers: SA, NSGA-II, SPEA2, MOACO, ALNS, MOPSO, GRASP and Gurobi.
- YAML-driven batch runner: parallel, checkpointed and resumable.
- Hypervolume and gap-to-best-known-solution reporting.
- Excel output with several sheets, plus route, convergence and Pareto plots.
- Slurm scripts for parameter tuning (hpc/).


INSTALLATION
------------
Python 3.10+ (developed on 3.13). From the repository root:

    python -m venv .venv
    source .venv/bin/activate        (Windows: .venv\Scripts\activate)
    pip install -e .

Alternatives: "pip install -r requirements.txt" or "uv sync".

Core dependencies: numpy, pandas, openpyxl, PyYAML, matplotlib, joblib.
The gurobi solver also needs gurobipy and a Gurobi licence
(pip install gurobipy). Nothing else needs it.


QUICK START
-----------
1. Instances

   Six instances are bundled in data/instances/ (C101, C201, R101, R201,
   RC101, RC201). To download more:

       python scripts/download_instances.py            (the six above)
       python scripts/download_instances.py --all      (all 56)
       python scripts/download_instances.py C202 R105  (specific ones)

2. Smoke test (finishes in seconds to about a minute)

       python scripts/run_experiment.py --config configs/smoke.yaml --plots

   Results appear in outputs/: smoke.xlsx, plots/, experiment.log and
   checkpoints_smoke/.

3. Real experiments

       python scripts/run_experiment.py --config configs/hard_tw.yaml
       python scripts/run_experiment.py --config configs/soft_tw.yaml

   configs/smoke.yaml       NSGA-II on C101, 2 runs, tiny budget
   configs/hard_tw.yaml     hard time windows, all 56 instances at 100
                            customers, six algorithms, 10 runs each
                            (set n_jobs to your core count)
   configs/soft_tw.yaml     soft time windows, 3 objectives, 25-customer
                            versions of C101, C202, R101, RC101
   configs/experiment.yaml  single-instance GRASP example

Command-line options for run_experiment.py:

   --config PATH   YAML or JSON config (required)
   --no-resume     ignore existing checkpoints and rerun everything
   --logfile PATH  log file (default outputs/experiment.log)
   --plots         also write PNG plots to outputs/plots/


LIBRARY USAGE
-------------
    import movrptw as mv

    ins = mv.parse_solomon("data/instances/C101.txt")

    # Single-objective: simulated annealing
    result = mv.get_solver("sa")(ins, ["num_vehicles", "total_distance"],
                                 {"iterations": 8000}, seed=1).run()
    print(result.best.objectives)
    print(result.best.routes)        # depot is implicit at both ends
    print(result.feasible)

    # Multi-objective: NSGA-II returns a Pareto front
    res = mv.get_solver("nsga2")(
        ins, ["total_distance", "num_vehicles"],
        {"tw_mode": "hard", "pop_size": 60, "generations": 100}, seed=1,
    ).run()
    for p in res.pareto:
        print(p.objectives, p.routes)

A result holds: best, pareto (non-dominated solutions, empty for SA),
runtime_s, iterations, convergence and feasible.


OBJECTIVES AND TIME WINDOWS
---------------------------
Objectives:

   num_vehicles      routes used
   total_distance    total route length, depot to depot
   total_wait        time spent waiting for customers to open
   total_tardiness   total lateness past customer due times
   route_imbalance   longest route distance minus shortest
   makespan          latest depot return time

Keep total_distance and num_vehicles in your objective list; gap
reporting and plotting use them.

Capacity and "each customer served exactly once" are always hard
constraints. Time windows depend on tw_mode:

   hard   a violation makes a solution infeasible.
          Objectives: [total_distance, num_vehicles]
   soft   a violation is allowed and measured.
          Objectives: [total_distance, num_vehicles, total_tardiness]

Hard mode must not include total_tardiness, and soft mode must include it;
anything else is rejected with an error. Set tw_mode explicitly for every
multi-objective algorithm in your config. SA has no tw_mode (it penalises
time-window violations), and Gurobi is always hard.

In the results, "feasible" (and success_rate) treats time windows as hard,
so in soft mode it means the reported best solution has zero tardiness.
"best" is the solution with the lowest scalarised score; use the pareto
data for the full trade-off.


CONFIGURATION
-------------
    output: outputs/results.xlsx      # workbook to write
    append: false                     # true = merge into existing workbook
    checkpoint_dir: outputs/checkpoints
    n_runs: 5                         # replications per configuration
    base_seed: 12345                  # run r uses base_seed + r
    n_jobs: 4                         # parallel workers (-1 = all cores)
    max_customers: null               # e.g. 25 or 50 = first N customers
    distance_rounding: null           # null | floor1 | round
    objectives: [total_distance, num_vehicles]

    instances:
      - data/instances/C101.txt
      - data/instances/R101.txt

    algorithms:
      - name: nsga2
        fixed: { tw_mode: hard }      # constant parameters
        grid:                         # every combination is run
          pop_size: [60, 100]
          generations: [150]
      - name: alns
        fixed: { tw_mode: hard, n_iterations: 3500, n_remove_frac: 0.15 }

Every (instance x algorithm x parameter set x seed) is one task. Finished
tasks are checkpointed, so an interrupted run resumes where it stopped.

Tips:
- max_customers keeps the first N customers of each instance. Best-known
  values are chosen by this size (25, 50 or 100).
- Use a separate output workbook and checkpoint_dir for each max_customers
  size, and a fresh checkpoint_dir if you change objectives or
  distance_rounding.
- Parameter names differ between solvers (for example iterations for SA,
  MOACO and GRASP, generations for NSGA-II and SPEA2, n_iterations for ALNS
  and MOPSO). See the list below.


SOLVERS
-------
   sa      Simulated annealing (single-objective)
   nsga2   NSGA-II with route crossover, route split and diversity control
   spea2   SPEA2 (strength and density fitness, truncated archive)
   moaco   Multi-objective ant colony, one pheromone matrix per objective
   alns    Adaptive large-neighbourhood search (random, worst, related and
           route-elimination destroy; greedy and regret repair)
   mopso   Discrete multi-objective particle swarm
   grasp   Randomised greedy construction plus local search
   gurobi  Exact epsilon-constraint MILP baseline

Parameters (defaults in brackets). Every solver also accepts "weights"
for the scalarisation used to pick the best solution.

   sa      iterations (8000), initial_temp (1.0), cooling (0.9995),
           restarts (1)
   nsga2   pop_size (100), generations (200), crossover_rate (1),
           mutation_rate (1), max_duplicates (3), tw_mode
   spea2   pop_size (60), archive_size (40), generations (150),
           crossover_rate (0.9), mutation_rate (0.3), k, tw_mode
   moaco   n_ants (30), iterations (100), rho (0.1), alpha (1.0),
           beta (2.0), tau0 (1.0), archive_size (100), tw_mode
   alns    n_iterations (5000), n_remove_frac (0.25), start_temp (100),
           cooling (0.9995), segment_size (100), tw_mode
   mopso   n_particles (50), n_iterations (300), max_archive (50), tw_mode
   grasp   iterations (300), alpha (0.3), local_search_max_passes (50),
           neighbor_k (12), time_limit (seconds), archive_max_size, tw_mode
   gurobi  time_limit (300 seconds per solve), threads (1)

MOACO gets slow on large instances, so test with a small run first.


OUTPUT WORKBOOK
---------------
   summary     per instance, algorithm and parameter set: mean/std/min/max
               of each objective, runtime, success_rate, mean/std hv and
               gap to best-known solution
   runs        one row per replication
   pareto      every point of every Pareto front, with its routes
   reference   the reference point and ideal/nadir used for hypervolume
   solutions   best routes of each run
   instances   instance data and best-known values
   config      high-level experiment settings

With append: true, new rows are added to the existing workbook and the
summary is recomputed.


HYPERVOLUME
-----------
Hypervolume (hv) measures how much objective space a Pareto front
dominates, up to a reference point. Higher is better. It rewards both
closeness to the optimum and spread, and does not need the true Pareto
front.

- Runs are compared only within the same instance and objective list, so
  hard and soft runs are never mixed.
- Objectives are min-max scaled using the group's pooled ideal and nadir,
  and the reference point is 1.1 in every scaled coordinate. HV is
  therefore comparable within an instance, not across instances.
- SA has no front, so its best solution counts as a one-point front.
- With append: true, rows already in the workbook keep their old HV
  values, so HV is only comparable within one pass. Rerun without append
  when HV matters.

Direct use:

    from movrptw import hypervolume, run_metrics
    hypervolume([[1., 3.], [2., 2.], [3., 1.]], ref_point=[4., 4.])   # 6.0
    quality, groups = run_metrics(results)


BEST-KNOWN SOLUTIONS
--------------------
movrptw/bks.py holds best-known (vehicles, distance) values for 25, 50 and
100 customers. The table is chosen from max_customers (100 if unset).
Distances are rounded to each table's precision before the gap is
computed. RC208 has no 50-customer entry, and unknown instances simply
give empty gaps. Add entries by extending the tables in that file.


GUROBI BASELINE
---------------
The gurobi solver solves the hard time-window problem exactly with an
epsilon-constraint sweep: first it minimises the vehicle count, then, for
each vehicle limit upward, it minimises distance. It is intended for the
25- and 50-customer versions (set max_customers) and uses distances
floored to one decimal at those sizes, matching the published best-known
tables.

scripts/rerun_gurobi_hard.py <25|50> re-solves the gurobi rows of an
existing results_hard.xlsx and merges routes, gaps, solver status and a
gurobi_sweep sheet back into it.


HPC (SLURM)
-----------
The hpc/ folder spreads a parameter grid over a Slurm job array, merges
the shards into one workbook and ranks the settings:

    python scripts/download_instances.py --all
    ./hpc/submit.sh hpc/specs/spea2.yaml --pilot     (time one shard)
    ./hpc/submit.sh hpc/specs/spea2.yaml             (full sweep)
    python hpc/rank.py <path>/tuning_spea2_hard.xlsx

See hpc/README.md for details. Account, partition and path settings in
the sbatch and shell scripts are for UCT's cluster; edit them for yours.


ADDING A SOLVER
---------------
Subclass MetaheuristicSolver, implement _solve, and register it:

    from movrptw.solution import Solution
    from movrptw.solvers.base import MetaheuristicSolver, register_solver
    from movrptw.solvers.construction import (
        bind_operators, nearest_neighbour_construct, random_neighbour)

    @register_solver("my_algo")
    class MyAlgo(MetaheuristicSolver):
        multi_objective = False       # True for Pareto-based algorithms

        def _solve(self) -> list[Solution]:
            ops = bind_operators(self.instance, self.dist)
            current = nearest_neighbour_construct(
                self.instance, self.dist, self.rng)
            for _ in range(int(self.params.get("iterations", 5000))):
                cand = random_neighbour(current, self.rng, ops)
                if self.scalar(self.evaluate(cand)) < \
                        self.scalar(self.evaluate(current)):
                    current = cand
                self._iterations += 1
                self._convergence.append(self.scalar(self.evaluate(current)))
            return [current]

Then use "my_algo" as an algorithm name in a config. Helpers available in
_solve: self.instance, self.dist, self.rng, self.np_rng, self.params,
self.evaluate(sol), self.score(sol), self.scalar(evaluation) and
self.objective_names.

- For a Pareto-based solver, subclass MOSolver (movrptw/solvers/mo_base.py),
  which adds tw_mode handling and a Pareto archive.
- To use a new solver with n_jobs above 1, import its module in
  movrptw/solvers/__init__.py.
- A new objective is added in movrptw/solution.py (_OBJECTIVE_ATTR).
- examples/custom_solver.py is a complete Tabu Search example.


PROJECT STRUCTURE
-----------------
    MO-VRP-Dev/
    +-- movrptw/
    |   +-- instance.py        Solomon parser, instance model
    |   +-- solution.py        solutions, route timing, objectives
    |   +-- tw.py              hard/soft time-window handling
    |   +-- config.py          config loading, grid expansion
    |   +-- experiment.py      batch runner, checkpoint/resume
    |   +-- results.py         aggregation, Excel writer
    |   +-- metrics.py         hypervolume
    |   +-- bks.py             best-known solutions
    |   +-- viz.py             plots
    |   +-- solvers/
    |       +-- base.py, mo_base.py, construction.py
    |       +-- sa.py, nsga2.py, spea2.py, moaco.py
    |       +-- alns.py, mopso.py, grasp.py, gurobi.py
    +-- configs/               smoke, experiment, hard_tw, soft_tw
    +-- data/instances/        bundled Solomon instances
    +-- scripts/               run_experiment.py, download_instances.py,
    |                          rerun_gurobi_hard.py, sbatch wrappers
    +-- hpc/                   Slurm tuning (make_configs, submit, rank)
    +-- examples/              custom_solver.py
    +-- pyproject.toml, requirements.txt, uv.lock


DATA ATTRIBUTION
----------------
Solomon benchmark instances are from M. M. Solomon (1987), "Algorithms for
the vehicle routing and scheduling problems with time window
constraints", Operations Research 35(2). Please cite Solomon and SINTEF
when using them.

Best-known values: 100 customers from the SINTEF benchmark; 25 customers
from Prag, Woolway and Jacobs (ISCMI 2019); 50 customers from Dorronsoro's
CVRPTW tables.


LICENSE
-------
MIT (as declared in pyproject.toml).
