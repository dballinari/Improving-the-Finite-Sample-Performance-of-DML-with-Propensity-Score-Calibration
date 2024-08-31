import pandas as pd
import numpy as np
from src import estimators, models
import random

PATH_TO_ALMP_DATA = "empirical_example\\Data\\1203_ALMP_Data_E_v1.0.0.csv"
NUM_REP = 10
RANDOM_STATE = 0
SAMPLE_FRACTIONS = [0.125, 0.25, 0.5, 1]

# Data========================================================================

data = pd.read_csv(PATH_TO_ALMP_DATA)

# Change sample such that it is the same as in Lechner, M., Knaus, M. C.,
# Huber, M., Frölich, M., Behncke, S., Mellace, G., & Strittmatter, A. (2020).
# Swiss Active Labor Market Policy Evaluation [Dataset]. Distributed by FORS,
# Lausanne. Retrieved from https://doi.org/10.23662/FORS-DS-1203-1)
df = data[(data.treatment6 == "no program") | (data.treatment6 == "language")].copy()


df["treated"] = np.where(df.treatment6 == "no program", 0, 1)


# Calculate the outcome of interest (first 6 months)

df["emp_1_6"] = df[
    ["employed1", "employed2", "employed3", "employed4", "employed5", "employed6"]
].sum(axis=1)

print(df.emp_1_6.value_counts())

df.drop(
    [f"employed{i}" for i in range(1, 37)],
    axis=1,
    inplace=True,
)


# Define covarites included in analysis
X_cols = [
    "age",
    "canton_french",
    "canton_german",
    "canton_italian",
    "city_big",
    "city_medium",
    "city_no",
    "cw_age",
    "cw_cooperative",
    "cw_educ_above_voc",
    "cw_educ_tertiary",
    "cw_female",
    "cw_id",
    "cw_missing",
    "cw_own_ue",
    "cw_tenure",
    "cw_voc_degree",
    "emp_share_last_2yrs",
    "emp_spells_5yrs",
    "employability",
    "female",
    "foreigner_b",
    "foreigner_c",
    "gdp_pc",
    "married",
    "other_mother_tongue",
    "past_income",
    "prev_job_manager",
    "prev_job_sec1",
    "prev_job_sec2",
    "prev_job_sec3",
    "prev_job_sec_mis",
    "prev_job_self",
    "prev_job_skilled",
    "prev_job_unskilled",
    "qual_degree",
    "qual_semiskilled",
    "qual_unskilled",
    "qual_wo_degree",
    "swiss",
    "ue_cw_allocation1",
    "ue_cw_allocation2",
    "ue_cw_allocation3",
    "ue_cw_allocation4",
    "ue_cw_allocation5",
    "ue_cw_allocation6",
    "ue_spells_last_2yrs",
    "unemp_rate",
]

df_treated = df[df["treated"] == 1].reset_index(drop=True)
index_treated = df_treated.index.tolist()

df_untreated = df[df["treated"] == 0].reset_index(drop=True)
index_untreated = df_untreated.index.tolist()

# Estimation==================================================================

results = {}

for estimator in ["RandomForest", "GradientBoosting", "Lasso"]:

    results_estimator = {}

    model_args = {"random_state": RANDOM_STATE}

    if estimator == "RandomForest":
        model_args = {"n_jobs": -1, **model_args}

    for fraction in SAMPLE_FRACTIONS:
        for rep_i in range(NUM_REP):
            if fraction == 1 and rep_i > 0:
                # for fraction=1 we use the entire sample, no need to repeate
                # the calculation multiple times
                results_estimator[fraction][:, :, 1:] = np.nan
                break

            # Define random state of current iteration
            random_state_rep = RANDOM_STATE + rep_i
            random.seed(random_state_rep)
            np.random.seed(random_state_rep)

            # Radnomly select treated and untreated observations
            sampled_index_treated = random.sample(
                index_treated, int(len(index_treated) * fraction)
            )
            sampled_index_untreated = random.sample(
                index_untreated, int(len(index_untreated) * fraction)
            )

            # Concatenate the two DataFrames
            sampled_df = pd.concat(
                [
                    df_treated.iloc[sampled_index_treated],
                    df_untreated.iloc[sampled_index_untreated],
                ]
            )

            # Shuffle the combined DataFrame
            sampled_df = sampled_df.sample(
                frac=1, random_state=random_state_rep
            ).reset_index(drop=True)

            # Tune hyperparameters
            hparams_reg_treated, hparams_reg_not_treated, hparams_propensity = (
                models.tune_nuisances(
                    sampled_df["emp_1_6"],
                    sampled_df["treated"],
                    sampled_df[X_cols],
                    nfolds=5,
                    model_name=estimator,
                    n_jobs_cv=-1,
                    **model_args,
                )
            )

            # estimate ATE
            estimation_results_trajectory, _, _ = estimators.estimate_ate(
                sampled_df["emp_1_6"].to_numpy(),
                sampled_df["treated"].to_numpy(),
                sampled_df[X_cols].to_numpy(),
                model_reg_treated=models.regression_model(
                    name=estimator, **model_args, **hparams_reg_treated
                ),
                model_reg_not_treated=models.regression_model(
                    name=estimator, **model_args, **hparams_reg_not_treated
                ),
                model_propensity=models.classification_model(
                    name=estimator, **model_args, **hparams_propensity
                ),
            )

            if results_estimator.get(fraction) is None:
                results_estimator[fraction] = np.zeros(
                    (
                        estimation_results_trajectory.shape[0],
                        estimation_results_trajectory.shape[1],
                        NUM_REP,
                    )
                )

            results_estimator[fraction][:, :, rep_i] = estimation_results_trajectory
    # store results for current estimator
    results[estimator] = results_estimator

# save results
np.savez_compressed("results/empirical_application.npz", **results)
