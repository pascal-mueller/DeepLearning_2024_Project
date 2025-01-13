#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pprint import pprint


def print_info(message):
    print(f"\033[92m[INFO]\033[0m {message}")  # Green


def print_warning(message):
    print(f"\033[93m[WARNING]\033[0m {message}")  # Yellow


def print_error(message):
    print(f"\033[91m[ERROR]\033[0m {message}")  # Red


def ensure_db_extension(dbname):
    """
    Ensures that the dbname ends with '.db'. If not, appends '.db'.

    Args:
        dbname (str): The original database name.

    Returns:
        str: The database name guaranteed to end with '.db'.
    """
    if not dbname.lower().endswith(".db"):
        dbname += ".db"
    return dbname


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run specified deep learning experiments with optional parameters or parameter search."
    )

    # Required argument for specifying the experiment name
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the experiment (e.g., 'tl_full_XOR', 'tl_light_XOR').",
    )

    # Required argument for specifying the experiment name
    parser.add_argument(
        "--run_name",
        required=True,
        help="Name of the experiment run (e.g., 'tl_full_XOR_run_1', 'tl_light_XOR_run_3'). This is used to save the results of the run in ./results/exp_name/run_name.",
    )

    # Create a mutually exclusive group for --params and --paramsearch,
    # allowing the user to provide either or neither
    exclusive_group = parser.add_mutually_exclusive_group()

    exclusive_group.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to JSON file with parameters for a single run (optional).",
    )

    # --paramsearch can optionally take a file
    exclusive_group.add_argument(
        "--paramsearch",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Run a parameter search. "
            "If given without a file, use default search space; "
            "if a file is specified, use that search config."
        ),
    )

    # Add flags for plotting
    parser.add_argument(
        "--plot_losses",
        action="store_true",
        help="Plot training losses (only valid if --paramsearch is not specified).",
    )
    parser.add_argument(
        "--plot_data",
        action="store_true",
        help="Plot data (only valid if --paramsearch is not specified).",
    )
    parser.add_argument(
        "--plot_fim",
        action="store_true",
        help="Plot Fisher Information Metric (only valid if --paramsearch is not specified).",
    )

    # The following arguments only make sense if paramsearch is used
    parser.add_argument(
        "--dbname",
        type=str,
        default=None,
        help="Optional DB name to store param search results (only if --paramsearch).",
    )
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of cores to use in param search (only if --paramsearch).",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to run in parameter search (only if --paramsearch).",
    )
    parser.add_argument(
        "--studyname",
        type=str,
        default=None,
        help="Name of the study (only if --paramsearch is used). Defaults to 'hyperparameter_optimization'.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validation Logic
    # ------------------------------------------------------------------

    # Enforce that --dbname, --num_cores, --num_trials, and --studyname
    # cannot be used unless --paramsearch is specified
    if not args.paramsearch and (
        args.dbname is not None
        or args.num_cores != 1
        or args.num_trials != 100
        or args.studyname is not None
    ):
        parser.error(
            "The arguments --dbname, --num_cores, --num_trials, and --studyname are only valid if --paramsearch is specified."
        )

    # Validate that plotting flags cannot be used with --paramsearch
    if args.paramsearch and (args.plot_losses or args.plot_data or args.plot_fim):
        parser.error(
            "The flags --plot_losses, --plot_data, and --plot_fim cannot be used together with --paramsearch."
        )

    # If --paramsearch is used and --dbname is not provided, set dbname to --name
    if args.paramsearch and args.dbname is None:
        args.dbname = args.name
        print_info(f"--dbname not provided. Using --name '{args.name}' as dbname.")

    # If --paramsearch is used and --studyname is not provided, set it to default
    if args.paramsearch and args.studyname is None:
        args.studyname = "hyperparameter_optimization"
        print_info(
            f"--studyname not provided. Using default study name '{args.studyname}'."
        )

    # Enforce that --studyname cannot be used without --paramsearch
    if args.studyname is not None and not args.paramsearch:
        parser.error(
            "The argument --studyname is only valid if --paramsearch is specified."
        )

    # Final Validation for --num_trials
    if args.paramsearch and args.num_trials <= 0:
        parser.error("--num_trials must be a positive integer.")

    if args.dbname is not None:
        args.dbname = ensure_db_extension(args.dbname)

    return args


def run_paramsearch(
    experiment_name: str,
    run_name,
    num_trials: int,
    num_cores: int,
    dbname: str,
    studyname: str,
):
    if experiment_name == "tl_full_XOR":
        from experiments.target_learning import full_XOR

        print(f"Running parameter search for experiment: {experiment_name} using")
        print(f" > Number of trials: {num_trials}")
        print(f" > Number of cores: {num_cores}")
        print(f" > DB Name: {dbname}")
        print(f" > Study Name: {studyname}")

        full_XOR.run_optuna_study(run_name, num_trials, num_cores, dbname, studyname)

        return

    if experiment_name == "bp_full_XOR":
        from experiments.backprop import full_XOR

        print(f"Running parameter search for experiment: {experiment_name} using")
        print(f" > Number of trials: {num_trials}")
        print(f" > Number of cores: {num_cores}")
        print(f" > DB Name: {dbname}")
        print(f" > Study Name: {studyname}")

        full_XOR.run_optuna_study(run_name, num_trials, num_cores, dbname, studyname)

        return

    print_error(f"Unknown experiment: {experiment_name}")

    return


def run_experiment(
    experiment_name: str,
    params: dict,
    run_name: str,
    plot_losses: bool = False,
    plot_data: bool = False,
    plot_fim: bool = False,
):
    """
    A wrapper to run the selected experiment.
    Adjust this to call whatever function your
    experiment script provides.
    """
    if params is not None:
        raise NotImplementedError(
            "params argument not implemented yet. You can go to the experiment in the experiments folder and replace BEST_PARAMS. Those are currently used when running the experiment."
        )

    #
    # TARGET LEARNING EXPERIMENTS
    #
    if experiment_name == "tl_full_XOR":
        from experiments.target_learning import full_XOR

        params = full_XOR.BEST_PARAMS
        print(f"Running experiment: {experiment_name} with params:\n")
        pprint(params)
        print("\n")
        _, perf, avg_perf = full_XOR.run_experiment(
            params,
            run_name=run_name,
            plot_data=plot_data,
            plot_losses=plot_losses,
            plot_fim=plot_fim,
            verbose_level=1,
        )
        print(f"\nAverage Performance: {avg_perf}")

        return

    #
    # BACKPROP EXPERIMENTS
    #
    if experiment_name == "bp_full_XOR":
        from experiments.backprop import full_XOR

        params = full_XOR.BEST_PARAMS
        print(f"Running experiment: {experiment_name} with params:\n")
        pprint(params)
        print("\n")
        _, perf, avg_perf = full_XOR.run_experiment(
            params,
            run_name=run_name,
            plot_data=plot_data,
            plot_losses=plot_losses,
            plot_fim=plot_fim,
            verbose_level=1,
        )
        print(f"\nAverage Performance: {avg_perf}")

        return

    print_error(f"Unknown experiment: {experiment_name}")

    return


if __name__ == "__main__":
    args = parse_arguments()

    # 2. If paramsearch is enabled, run the search, else run a single experiment
    if args.paramsearch:
        run_paramsearch(
            args.name,
            args.run_name,
            args.num_trials,
            args.num_cores,
            args.dbname,
            args.studyname,
        )
    else:
        run_experiment(
            args.name,
            args.params,
            args.run_name,
            args.plot_losses,
            args.plot_data,
            args.plot_fim,
        )
