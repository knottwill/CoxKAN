"""
Hyperparameter searching.
"""

import os
import optuna
from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from optuna.pruners import MedianPruner
from optuna.trial import Trial
import torchtuples as tt
import torch
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

from . import CoxKAN
from .utils import FastCoxLoss

class Sweep:
    """ Hyperparameter search (sweep) object. """

    def __init__(self, model='coxkan', search_config=None):
        """
        Args:
        -----
            model: str
                'coxkan' or 'mlp' ('deepsurv' is the same as 'mlp').
            search_config: str
                Path to the hyperparameter search space configuration file. If None, default search space is used.
        """
        if model == 'deepsurv': model = 'mlp'
        assert model in ['coxkan', 'mlp'], f"Model {model} not supported."
        self.model = model
        if search_config is None:
            path = Path(__file__).resolve()
            self.search_space = yaml.safe_load(open(path.parent / f'_configs/{model}_sweep.yml', 'r'))
        else:
            self.search_space = yaml.safe_load(open(search_config, 'r'))

    def run_cv(self, df, duration_col, event_col, study_name=None, storage=None, n_trials=100, 
            n_folds=3, folds=None, save_params=None, n_jobs=1, verbose=1, seed=None):
        """ Run sweep based on cross validation of df. 
        
        Args:
        -----
            df: pd.DataFrame
                DataFrame with duration and event columns.
            duration_col: str
                Name of the duration column.
            event_col: str
                Name of the event column.
            study_name: str
                Name of the optuna study.
            storage: str
                Path to the optuna study storage.
            n_trials: int
                Number of trials to run.
            n_folds: int
                Number of folds for cross-validation.
            folds: list
                List of tuples of train-test indices for each fold.
            save_params: str
                Path to save the best hyperparameters.
            n_jobs: int
                Number of parallel jobs.
            verbose: int
                Verbosity level.
            seed: int
                Random seed.

        Returns:
        --------
            study: optuna.study
                Optuna study object.
        """
        self.df = df
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariates = df.columns.difference([duration_col, event_col])
        self.verbose = verbose

        if n_jobs > 1 and self.model == 'coxkan':
            print("Warning: n_jobs > 1 for CoxKAN not yet supported -> setting n_jobs=1. Please use the storage option for parallelization.")
            n_jobs = 1

        if folds is None:
            skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
            self.folds = list(skf.split(df, df[event_col]))
        else:
            assert len(folds) == n_folds
            self.folds = folds

        study = optuna.create_study(
            study_name=study_name, storage=storage, load_if_exists=True,
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed), pruner=MedianPruner()
        )

        objective = self._cv_objective_coxkan if self.model=='coxkan' else self._cv_objective_mlp
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        if save_params is not None:
            save_params = Path(save_params)
            save_params.parent.mkdir(parents=True, exist_ok=True)
            with open(save_params, 'w') as f:
                yaml.dump(self._make_config(study.best_params), f)
        
        return study
    
    def run_val(self, df_train, df_val, duration_col, event_col, study_name=None, storage=None, 
               n_trials=100, n_runs_per_trial=1, save_params=None, n_jobs=1, verbose=1, seed=None):
        """ Run sweep based on a single validation set. 
        
        Args:
        -----
            df_train: pd.DataFrame
                DataFrame with duration and event columns for training.
            df_val: pd.DataFrame
                DataFrame with duration and event columns for validation.
            duration_col: str
                Name of the duration column.
            event_col: str
                Name of the event column.
            study_name: str
                Name of the optuna study.
            storage: str
                Path to the optuna study storage.
            n_trials: int
                Number of trials to run.
            n_runs_per_trial: int
                Number of runs per trial (i.e. the number of times to evaluate each set of hyperparameters)
            save_params: str
                Path to save the best hyperparameters.
            n_jobs: int
                Number of parallel jobs.
            verbose: int
                Verbosity level.
            seed: int
                Random seed.

        Returns:
        --------
            study: optuna.study
                Optuna study object.
        """

        self.df_train = df_train
        self.df_val = df_val
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariates = df_train.columns.difference([duration_col, event_col])
        self.verbose = verbose
        self.n_runs_per_trial = n_runs_per_trial

        if n_jobs > 1 and self.model == 'coxkan':
            print("Warning: n_jobs > 1 for CoxKAN not yet supported -> setting n_jobs=1. Please use the storage option for parallelization.")
            n_jobs = 1

        study = optuna.create_study(
            study_name=study_name, storage=storage, load_if_exists=True,
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed), pruner=MedianPruner()
        )

        objective = self._val_objective_coxkan if self.model=='coxkan' else self._val_objective_mlp
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        if save_params is not None:
            save_params = Path(save_params)
            save_params.parent.mkdir(parents=True, exist_ok=True)
            with open(save_params, 'w') as f:
                yaml.dump(self._make_config(study.best_params), f)

        return study

    def _suggest_params(self, trial: Trial, search_space: dict):
        ### Optuna suggest parameters from search space
        params = {}
        for param, conf in search_space.items():
            if conf["type"] == "categorical":
                params[param] = trial.suggest_categorical(param, conf["values"])
            elif conf["type"] == "int":
                params[param] = trial.suggest_int(param, conf["min"], conf["max"])
            elif conf["type"] == "float":
                params[param] = trial.suggest_float(param, conf["min"], conf["max"])
            elif conf["type"] == "loguniform":
                params[param] = trial.suggest_float(param, conf["min"], conf["max"], log=True)
            elif conf["type"] == "fixed":
                # Using suggest categorical so they appear in best_params
                params[param] = trial.suggest_categorical(param, [conf["value"]])

        return params
        
    def _make_config(self, params):
        ### Make config from suggested parameters
        if 'early_stopping' in params.keys() and params['early_stopping']:
            if 'steps' in params:
                params['steps'] = 300
            if 'epochs' in params:
                params['epochs'] = 300

        if self.model == 'coxkan':
            init_params = ['grid', 'k', 'base_fun', 'noise_scale', 'noise_scale_base', 'symbolic_enabled', 'bias_trainable', 'grid_eps', 'grid_range', 'sp_trainable', 'sb_trainable']
            train_params = ['lr', 'steps', 'early_stopping', 'lamb', 'lamb_entropy', 'lamb_coef']
            config = {'init_params': {}, 'train_params': {}, 'prune_threshold': 0.0}

            config['init_params']['width'] = [len(self.covariates), *[params['hidden_dim'] for _ in range(params['num_hidden'])], 1]

            for param, value in params.items():
                if param in init_params:
                    config['init_params'][param] = value
                elif param in train_params:
                    config['train_params'][param] = value
                elif param == 'prune_threshold':
                    config['prune_threshold'] = value

        elif self.model == 'mlp':
            init_params = ['num_nodes', 'batch_norm', 'dropout']
            optimizer_params = ['lr', 'weight_decay']
            config = {'init_params': {}, 'optimizer_params': {}, 'early_stopping': False, 'epochs': 100}

            config['init_params']['num_nodes'] = [len(self.covariates), *[params['hidden_dim'] for _ in range(params['num_hidden'])], 1]

            for param, value in params.items():
                if param in init_params:
                    config['init_params'][param] = value
                elif param in optimizer_params:
                    config['optimizer_params'][param] = value
                elif param == 'early_stopping':
                    config['early_stopping'] = value
                elif param == 'epochs':
                    config['epochs'] = value

            if config['early_stopping']:
                config['epochs'] = 300

        return config
    
    def _cv_objective_coxkan(self, trial: Trial):
        ### Cross-validation objective function for CoxKAN
        config = self._make_config(self._suggest_params(trial, self.search_space))

        cindices = []
        for k, (train_idx, test_idx) in enumerate(self.folds):
            train = self.df.iloc[train_idx]
            test = self.df.iloc[test_idx]
            
            model = CoxKAN(**config['init_params'])
            if 'early_stopping' in config['train_params'] and config['train_params']['early_stopping']:
                train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train[self.event_col])
                model.train(train, val, duration_col=self.duration_col, event_col=self.event_col, progress_bar=False, **config['train_params'])
            else:
                model.train(train, duration_col=self.duration_col, event_col=self.event_col, progress_bar=False, **config['train_params'])
            model.prune_edges(threshold=config['prune_threshold'], verbose=False)

            cindex = model.cindex(test)
            cindices.append(cindex)
            if self.verbose: print(f'Fold {k} c-index: {cindex}')

            trial.report(cindex, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cindices)
    
    def _val_objective_coxkan(self, trial: Trial):
        ### Single validation set objective function for CoxKAN
        config = self._make_config(self._suggest_params(trial, self.search_space))

        cindices = []
        for k in range(self.n_runs_per_trial):
            model = CoxKAN(seed=k, **config['init_params'])
            if 'early_stopping' in config['train_params'] and config['train_params']['early_stopping']:
                train, val = train_test_split(self.df_train, test_size=0.2, random_state=k, stratify=self.df_train[self.event_col])
                model.train(train, val, duration_col=self.duration_col, event_col=self.event_col, progress_bar=False, **config['train_params'])
            else:
                model.train(self.df_train, duration_col=self.duration_col, event_col=self.event_col, progress_bar=False, **config['train_params'])
            model.prune_edges(threshold=config['prune_threshold'], verbose=False)

            cindex = model.cindex(self.df_val)
            cindices.append(cindex)
            if self.verbose: print(f'Run {k} c-index: {cindex}')

            if self.n_runs_per_trial > 1:
                trial.report(cindex, k)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return np.mean(cindices)
    
    def _cv_objective_mlp(self, trial: Trial):
        ### Cross-validation objective function for MLP (DeepSurv)
        config = self._make_config(self._suggest_params(trial, self.search_space))

        cindices = []
        for k, (train_idx, test_idx) in enumerate(self.folds):
            train = self.df.iloc[train_idx]
            test = self.df.iloc[test_idx]

            if config['early_stopping']:
                train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train[self.event_col])

            mlp = tt.practical.MLPVanilla(
                in_features=len(self.covariates), out_features=1, output_bias=False, **config['init_params']
            )
            optimizer = tt.optim.Adam(**config['optimizer_params'])
            model = tt.Model(mlp, loss=FastCoxLoss, optimizer=optimizer)

            # Convert to PyTorch tensors
            X_train = torch.tensor(train[self.covariates].values).double()
            y_train = torch.tensor(train[[self.duration_col, self.event_col]].values).double()
            X_test = torch.tensor(test[self.covariates].values).double()
            y_test = torch.tensor(test[[self.duration_col, self.event_col]].values).double()

            if config['early_stopping']:
                X_val = torch.tensor(val[self.covariates].values).double()
                y_val = torch.tensor(val[[self.duration_col, self.event_col]].values).double()

                log = model.fit(
                    X_train, y_train, batch_size=len(X_train), val_data=(X_val, y_val), epochs=config['epochs'], verbose=False,
                    callbacks=[tt.callbacks.EarlyStopping(patience=20)]
                )

            else:
                log = model.fit(
                    X_train, y_train, batch_size=len(X_train), val_data=(X_test, y_test), epochs=config['epochs'], verbose=False
                )

            # Evaluation
            lph = model.predict(X_test)
            cindex = concordance_index(test[self.duration_col], -lph, test[self.event_col])
            cindices.append(cindex)
            if self.verbose: print(f'Fold {k} c-index: {cindex}')

            trial.report(cindex, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cindices)

    def _val_objective_mlp(self, trial: Trial):
        ### Single validation set objective function for MLP (DeepSurv)
        config = self._make_config(self._suggest_params(trial, self.search_space))

        cindices = []
        for k in range(self.n_runs_per_trial):

            if config['early_stopping']:
                train, val = train_test_split(self.df_train, test_size=0.2, random_state=k, stratify=self.df_train[self.event_col])

            mlp = tt.practical.MLPVanilla(
                in_features=len(self.covariates), out_features=1, output_bias=False, **config['init_params']
            )
            optimizer = tt.optim.Adam(**config['optimizer_params'])
            model = tt.Model(mlp, loss=FastCoxLoss, optimizer=optimizer)

            # Convert to PyTorch tensors
            X_train = torch.tensor(self.df_train[self.covariates].values).double()
            y_train = torch.tensor(self.df_train[[self.duration_col, self.event_col]].values).double()
            X_val = torch.tensor(self.df_val[self.covariates].values).double()
            y_val = torch.tensor(self.df_val[[self.duration_col, self.event_col]].values).double()

            if config['early_stopping']:
                log = model.fit(
                    X_train, y_train, batch_size=len(X_train), val_data=(X_val, y_val), epochs=config['epochs'], verbose=False,
                    callbacks=[tt.callbacks.EarlyStopping(patience=20)]
                )

            else:
                log = model.fit(
                    X_train, y_train, batch_size=len(X_train), val_data=(X_val, y_val), epochs=config['epochs'], verbose=False
                )

            # Evaluation
            lph = model.predict(X_val)
            cindex = concordance_index(self.df_val[self.duration_col], -lph, self.df_val[self.event_col])
            cindices.append(cindex)
            if self.verbose: print(f'Run {k} c-index: {cindex}')

            if self.n_runs_per_trial > 1:
                trial.report(cindex, k)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return np.mean(cindices)