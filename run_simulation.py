from sklearnex import patch_sklearn

patch_sklearn()

import argparse
import numpy as np
from tqdm import tqdm
import os

from src import data, estimators, models

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--num_simulations", type=int, default=1000)
argparser.add_argument("--n", type=int, default=1000)
argparser.add_argument("--p", type=int, default=30)
argparser.add_argument("--sigma", type=int, default=1)
argparser.add_argument("--mode", type=int, default=1)
argparser.add_argument("--n_folds", type=int, default=2)
argparser.add_argument("--model_name", type=str, default="RandomForest")
argparser.add_argument("--seed", type=int, default=256)
argparser.add_argument("--n_jobs", type=int, default=-1)
argparser.add_argument("--num_simulations_hyperparam", type=int, default=20)
args = argparser.parse_args()

model_args = {"random_state": args.seed}

if args.model_name == "RandomForest":
    model_args = {"n_jobs": args.n_jobs, **model_args}

# # set seed for hyperparameter tuning
np.random.seed(args.seed)
# list to store optimal hyperparameters
hparams_reg_treated = []
hparams_reg_not_treated = []
hparams_propensity = []
# add progress bar
progress_bar = tqdm(total=args.num_simulations_hyperparam, desc="Hyperparameter tuning")
for i in range(args.num_simulations_hyperparam):
    y, x, w, _, _, _, _ = data.simulate_data(
        n=args.n, p=args.p, mode=args.mode, sigma=args.sigma
    )
    hparams_reg_treated_i, hparams_reg_not_treated_i, hparams_propensity_i = (
        models.tune_nuisances(
            y,
            w,
            x,
            nfolds=args.n_folds,
            model_name=args.model_name,
            n_jobs_cv=args.n_jobs,
            **model_args,
        )
    )
    hparams_reg_treated.append(hparams_reg_treated_i)
    hparams_reg_not_treated.append(hparams_reg_not_treated_i)
    hparams_propensity.append(hparams_propensity_i)

    progress_bar.update(1)

progress_bar.close()

# find most frequent hyperparameters
best_hparams_reg_treated = models.modes_of_values(hparams_reg_treated)
best_hparams_reg_not_treated = models.modes_of_values(hparams_reg_not_treated)
best_hparams_propensity = models.modes_of_values(hparams_propensity)

results_estimates = None
seeds_simulations = args.seed + np.arange(args.num_simulations)
# add progress bar
progress_bar = tqdm(total=args.num_simulations, desc="Simulations")
for i in range(args.num_simulations):
    # set seed
    np.random.seed(seeds_simulations[i])

    while True:
        y, x, w, tau, b, e, ate = data.simulate_data(
            n=args.n, p=args.p, mode=args.mode, sigma=args.sigma
        )
        if np.sum(w) > 0 and np.sum(1 - w) > 0:
            break

    # estimate ATE
    estimation_results_trajectory, approach_names, evaluation_names = (
        estimators.estimate_ate(
            y,
            w,
            x,
            model_reg_treated=models.regression_model(
                name=args.model_name, **model_args, **best_hparams_reg_treated
            ),
            model_reg_not_treated=models.regression_model(
                name=args.model_name, **model_args, **best_hparams_reg_not_treated
            ),
            model_propensity=models.classification_model(
                name=args.model_name, **model_args, **best_hparams_propensity
            ),
            e=e,
            nfolds=args.n_folds,
        )
    )
    # create results array if it does not exist: num. simulations x num. methods x num. metrics
    if results_estimates is None:
        results_estimates = np.zeros(
            (args.num_simulations, *estimation_results_trajectory.shape)
        )
    results_estimates[i] = estimation_results_trajectory

    progress_bar.update(1)

progress_bar.close()


# approximate true ATE
_, _, _, _, _, _, ate = data.simulate_data(
    n=10**6, p=args.p, mode=args.mode, sigma=args.sigma
)

# create results folder if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")
# define file name from input arguments
args_list = list(vars(args).items())
file_name = "__".join([f"{k}{v}" for k, v in args_list])

np.savez(
    f"results/{file_name}.npz",
    approaches=approach_names,
    evaluations=evaluation_names,
    results_estimates=results_estimates,
    seeds_simulations=seeds_simulations,
    true_ate=ate,
    simulation_settings=vars(args),
    best_hparams_reg_treated=best_hparams_reg_treated,
    best_hparams_reg_not_treated=best_hparams_reg_not_treated,
    best_hparams_propensity=best_hparams_reg_treated,
)
