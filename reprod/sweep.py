"""
Script to run hyperparameter search ('sweep') for CoxKAN and MLP models on a given dataset.

We perform the sweep by optimizing the average C-Index from a cross-validation of the training set. 

Usage:
python reprod/sweep.py --exp_name <experiment_name> --model <model>

experiment_name can be: nwtco | flchain | support | metabric | gbsg | sim_gaussian | sim_depth_1 | sim_difficult | sim_deep
or TCGA-<cohort> where cohort: BRCA | STAD | GBMLGG | KIRC
"""

import sys 
from pathlib import Path
import argparse
import yaml
import pickle
import numpy as np
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
import pandas as pd

# add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(project_root))

from coxkan.datasets import nwtco, flchain, support, metabric, gbsg, create_dataset
from coxkan import CoxKAN
from coxkan.hyperparam_search import Sweep

parser = argparse.ArgumentParser(description="Cross-validation hyperparameter search ('sweep') for CoxKAN and MLP models.")
parser.add_argument('--exp_name', type=str, help='experiment name: nwtco | flchain | support | metabric | gbsg | sim_gaussian | sim_depth_1 | sim_difficult | TCGA-<proj>_<modality> (proj: BRCA, STAD, GBMLGG, KIRC | modality: clin | hist | gen | multi)')
parser.add_argument('--model', type=str, default='coxkan', help='model to sweep: coxkan | mlp')
parser.add_argument('--storage', type=str, default='sqlite:///./checkpoints/optuna.db', help='storage URL for optuna study')
parser.add_argument('--search_config', type=str, default=None, help='config file of hyperparameter search space (leave None for default)')
parser.add_argument('--n_trials', type=int, default=100, help='number of trials for sweep')
parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
parser.add_argument('--n_folds', type=int, default=4, help='number of folds for cross-validation')
parser.add_argument('--configs_dir', type=str, default='./configs', help='directory to save hyperparameter configs')
parser.add_argument('--seed', type=int, default=None, help='random seed - though this does not seem to be making it deterministic currently')
args = parser.parse_known_args()[0]

if args.seed is None: args.seed = np.random.randint(0, 1000)

if args.storage == "None": args.storage = None

datasets = {
    'nwtco': nwtco,
    'flchain': flchain,
    'support': support,
    'metabric': metabric,
    'gbsg': gbsg,
}

### Real datasets of simple covariates
if args.exp_name in datasets:
    dataset = datasets[args.exp_name]
    df_train, df_test = dataset.load(split=True)
    exp_name, duration_col, event_col, covariates = dataset.metadata()
    df_train, df_test = CoxKAN(width=[1]).process_data(df_train, df_test, duration_col, event_col)

    if args.model == "mlp" and args.exp_name not in ['nwtco', 'flchain']:
        print(f"Note: Sweeping mlp for {args.exp_name} is un-necessary, as we have results from DeepSurv publication.")

### Simulated datasets
elif args.exp_name[:4] == 'sim_':
    sim_name = args.exp_name[4:]

    with open(f'./configs/simulation/{sim_name}.yml', 'r') as file:
        sim_config = yaml.safe_load(file)
        sim_config['log_partial_hazard'] = eval(sim_config['log_partial_hazard'])

    df = create_dataset(
        sim_config['log_partial_hazard'],      # true log-partial hazard function
        baseline_hazard=sim_config['baseline_hazard'], # baseline hazard
        ranges=sim_config['ranges'],  # ranges of the covariates
        n_samples=sim_config['n_samples'],  # number of samples
        seed=args.seed)
    duration_col, event_col = "duration", "event"
    covariates = df.columns.drop([duration_col, event_col])

    # stratified split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['event'])

    # save train/test data for reproducibility
    df_train.to_csv(f'./data/{args.exp_name}_train.csv', index=False)
    df_test.to_csv(f'./data/{args.exp_name}_test.csv', index=False)

### TCGA cohorts (multi-modal data)
elif args.exp_name[:5] == 'TCGA-':

    cohort = args.exp_name[5:]

    df_train = pd.read_csv(f'data/TCGA/{cohort}_train.csv', index_col=0)
    df_test = pd.read_csv(f'data/TCGA/{cohort}_test.csv', index_col=0)
    duration_col, event_col = 'duration', 'event'
    covariates = [col for col in df_train.columns if col not in [duration_col, event_col]]

cv_search = Sweep(model=args.model, search_config=args.search_config)

study = cv_search.run_cv(df_train, 
                        duration_col, 
                        event_col, 
                        study_name=args.exp_name if args.storage is not None else None,
                        storage=args.storage,
                        n_trials=args.n_trials, 
                        n_folds=args.n_folds, 
                        save_params=Path(args.configs_dir) / f'{args.model}/{args.exp_name}.yml',
                        n_jobs=args.n_jobs, 
                        verbose=False,
                        seed=args.seed)

pruned_trials = [trial for trial in study.trials if trial.state == TrialState.PRUNED]
print(f"Number of pruned trials: {len(pruned_trials)}/{args.n_trials}")

# save study object
with open(Path(args.configs_dir) / f'{args.model}/{args.exp_name}_study.pkl', 'wb') as f:
    pickle.dump(study, f)
    