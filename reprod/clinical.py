"""
Usage:
python reprod/clinical.py --exp_name <exp_name> [--seed <seed>]

where --exp_name can be: gbsg | metabric | flchain

Evaluation on the other clinical datasets (support, nwtco) involve additional interpretability which
is better suited for the notebook interface.
"""

import numpy as np
import torch
import argparse
import sys
import pickle
import pandas as pd
from lifelines import CoxPHFitter
import torchtuples as tt
import yaml 
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import uuid
import warnings

# add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(project_root))

from coxkan import CoxKAN
from coxkan.datasets import gbsg, metabric, support, nwtco, flchain
from coxkan.utils import FastCoxLoss, count_parameters, bootstrap_metric, set_seed, SYMBOLIC_LIB

parser = argparse.ArgumentParser(description="Cross-validation hyperparameter search ('sweep') for CoxKAN and MLP models.")
parser.add_argument('--exp_name', type=str, help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='')
parser.add_argument('--no_symbolic', action='store_true', help='do not perform symbolic fitting')
args = parser.parse_known_args()[0]

# set random seed
if args.seed is not None:
    SEED = args.seed
else:
    np.random.seed(None)
    SEED = np.random.randint(0, 100000)
print('SEED: ', set_seed(SEED))

###############
### Load data
###############

datasets = {'gbsg': gbsg, 'metabric': metabric, 'flchain': flchain}
assert args.exp_name in datasets, f"Invalid experiment name: {args.exp_name}"

output_dir = Path('checkpoints') / args.exp_name
output_dir.mkdir(parents=True, exist_ok=True)

# load dataset
dataset = datasets[args.exp_name]
df_train, df_test = dataset.load(split=True)
dataset_name, duration_col, event_col, covariates = dataset.metadata()
assert dataset_name == args.exp_name

# FLCHAIN: changing sample.yr and flc.grp to float
if args.exp_name == 'flchain':
    df_train['sample.yr'] = df_train['sample.yr'].astype(float)
    df_train['flc.grp'] = df_train['flc.grp'].astype(float)
    df_test['sample.yr'] = df_test['sample.yr'].astype(float)
    df_test['flc.grp'] = df_test['flc.grp'].astype(float)
    run_deepsurv = True 
else:
    run_deepsurv = False

results = {'CoxKAN': {}}
if run_deepsurv: results['DeepSurv'] = {}

### Loading configs

with open(f'configs/coxkan/{args.exp_name}.yml', 'r') as f:
    config = yaml.safe_load(f)

with open(output_dir / 'config.yml', 'w') as f:
    yaml.dump(config, f)

if 'early_stopping' not in config['train_params']:
    config['train_params']['early_stopping'] = False

if run_deepsurv:
    with open(f'configs/mlp/{args.exp_name}.yml', 'r') as f:
        mlp_config = yaml.safe_load(f)

    if 'early_stopping' not in mlp_config:
        mlp_config['early_stopping'] = False

### Init CoxKAN
ckan = CoxKAN(seed=SEED, **config['init_params'])
coxkan_params = count_parameters(ckan)

# for the simple datasets, we first pre-process and register the data with the CoxKAN object
if 'TCGA' not in args.exp_name:
    df_train, df_test = ckan.process_data(df_train, df_test, duration_col, event_col, normalization='standard')

# if early stopping, split the training data into train and validation sets
if config['train_params']['early_stopping'] or (run_deepsurv and mlp_config['early_stopping']):
    train, val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train[event_col])

################
###  CoxPH
################
try:
    run_coxph = True
    cph = CoxPHFitter()
    cph.fit(df_train, duration_col=duration_col, event_col=event_col)
    def cph_cindex(df):
        return cph.score(df, scoring_method='concordance_index')
    cindex_train = bootstrap_metric(cph_cindex, df_train, N=100)['formatted']
    cindex_test = bootstrap_metric(cph_cindex, df_test, N=100)['formatted']
    coxph_str = f'CoxPH - train: {cindex_train}, test: {cindex_test}'; print(coxph_str)
    with open(output_dir / 'cindex.txt', 'w') as f:
        f.write(coxph_str + '\n')
    results['CoxPH'] = {'train': cindex_train, 'test': cindex_test, 'summary': cph.summary}
except Exception as e:
    run_coxph = False
    print('CoxPH failed:', e)

################
###  DeepSurv
################

if run_deepsurv:

    mlp = tt.practical.MLPVanilla(
        in_features=len(covariates), out_features=1, output_bias=False, **mlp_config['init_params']
    )
    optimizer = tt.optim.Adam(**mlp_config['optimizer_params'])
    deepsurv = tt.Model(mlp, loss=FastCoxLoss, optimizer=optimizer)
    deepsurv_params = count_parameters(mlp)

    # Convert to PyTorch tensors
    X_test = torch.tensor(df_test[covariates].values).double()
    y_test = torch.tensor(df_test[[duration_col, event_col]].values).double()

    def mlp_cindex(df):
        lph = deepsurv.predict(torch.tensor(df[covariates].values).double())
        return concordance_index(df[duration_col], -lph, df[event_col])

    def mlp_cindex_metric_fn(lph, labels):
        return concordance_index(labels[:, 0].detach().numpy(), -lph.detach().numpy(), labels[:, 1].detach().numpy())

    # Training
    if mlp_config['early_stopping']:
        X_val = torch.tensor(val[covariates].values).double()
        y_val = torch.tensor(val[[duration_col, event_col]].values).double()
        X_train = torch.tensor(train[covariates].values).double()
        y_train = torch.tensor(train[[duration_col, event_col]].values).double()
        log = deepsurv.fit(
            X_train, y_train, batch_size=len(X_train), val_data=(X_val, y_val), epochs=mlp_config['epochs'], verbose=False,
            metrics={'cindex': mlp_cindex_metric_fn}, callbacks=[tt.callbacks.EarlyStopping(patience=20)]
        )
    else:
        X_train = torch.tensor(df_train[covariates].values).double()
        y_train = torch.tensor(df_train[[duration_col, event_col]].values).double()
        log = deepsurv.fit(
            X_train, y_train, batch_size=len(X_train), val_data=(X_test, y_test), epochs=mlp_config['epochs'], verbose=False,
            metrics={'cindex': mlp_cindex_metric_fn}
        )
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(log.to_pandas()['train_loss'], label='train'); ax[0].plot(log.to_pandas()['val_loss'], label='val')
    ax[1].plot(log.to_pandas()['train_cindex'], label='train'); ax[1].plot(log.to_pandas()['val_cindex'], label='val')

    # put a vertical line at the best epoch
    if mlp_config['early_stopping']:
        best_epoch = log.to_pandas().val_cindex.idxmax()
        ax[0].axvline(best_epoch, color='k', linestyle='--', label='best model')
        ax[1].axvline(best_epoch, color='k', linestyle='--', label='best model')

    ax[0].legend(); ax[0].set_title('Loss'); ax[1].legend(); ax[1].set_title('C-index')
    fig.savefig(output_dir / 'mlp_training.png')

    cindex_train = bootstrap_metric(mlp_cindex, df_train, N=100)['formatted']
    cindex_test = bootstrap_metric(mlp_cindex, df_test, N=100)['formatted']

    deepsurv_str = f'DeepSurv - train: {cindex_train}, test: {cindex_test}'; print(deepsurv_str)
    with open(output_dir / 'cindex.txt', 'a') as f:
        f.write(deepsurv_str + '\n')
    results['DeepSurv'] = {'train': cindex_train, 'test': cindex_test, 'n_params': deepsurv_params}

##############
### CoxKAN
##############

### Training
if config['train_params']['early_stopping']:
    # config['train_params']['steps'] = 500 # with early stopping we are free to choose high number of steps
    log = ckan.train(train, val, duration_col, event_col, **config['train_params'])
else:
    log = ckan.train(df_train, df_test, duration_col, event_col, **config['train_params'])

_ = ckan.predict(df_test)
ckan.save_ckpt(output_dir / 'model.pt')
fig = log.plot()
fig.savefig(output_dir / 'coxkan_training.png')

cindex_train = bootstrap_metric(ckan.cindex, df_train, N=100)['formatted']
cindex_val = bootstrap_metric(ckan.cindex, val, N=100)['formatted'] if config['train_params']['early_stopping'] else None
cindex_test = bootstrap_metric(ckan.cindex, df_test, N=100)['formatted']

ckan_pre_str = f'CoxKAN - train: {cindex_train}, test: {cindex_test}'; print(ckan_pre_str)
with open(output_dir / 'cindex.txt', 'a') as f:
    f.write(ckan_pre_str + '\n')
results['CoxKAN']['Pre'] = {'train': cindex_train, 'test': cindex_test, 'val': cindex_val, 'n_params': coxkan_params}

fig = ckan.plot(beta=10, folder=f'./activations/{args.exp_name}{args.id}_pre')
fig.savefig(output_dir / 'coxkan_pre.png')

# save results
with open(output_dir / 'results.pkl', 'wb') as f:
    pickle.dump(results, f)
    print('Results saved to', output_dir / 'results.pkl')

if args.no_symbolic:
    sys.exit()

### Pruning

# If early stopping was used, we have a validation set to select the best pruning threshold
if config['train_params']['early_stopping']:
    pruning_thresholds = np.linspace(0, 0.05, 20)
    pruning_thresholds[0] = config['prune_threshold']
    cindices = []
    for threshold in pruning_thresholds:
        ckan_ = CoxKAN(seed=SEED, **config['init_params'])
        ckan_.load_ckpt(output_dir / 'model.pt', verbose=False)
        _ = ckan_.predict(df_test)
        
        prunable = True
        for l in range(ckan_.depth):
            if not (ckan_.acts_scale[l] > threshold).any():
                prunable = False
                break
        if not prunable:
            if threshold == config['prune_threshold']: continue
            else: break
            
        ckan_ = ckan_.prune_nodes(threshold)

        if 0 in ckan_.width: prunable = False
        if not prunable:
            if threshold == config['prune_threshold']: continue
            else: break

        _ = ckan_.predict(df_test)
        ckan_.prune_edges(threshold, verbose=False)
        cindices.append(ckan_.cindex(val))
        print(f'Pruning threshold: {threshold:.2f}, C-Index (Val): {cindices[-1]:.6f}')
    best_threshold = pruning_thresholds[np.argmax(cindices)]
    if np.max(cindices) < 0.51: best_threshold = 0
else:
    best_threshold = config['prune_threshold']
print(f'Best pruning threshold: {best_threshold}')

results['prune_threshold'] = best_threshold
_ = ckan.predict(df_test)
ckan = ckan.prune_nodes(best_threshold)
_ = ckan.predict(df_test)
ckan.prune_edges(best_threshold, verbose=True)
_ = ckan.predict(df_test)

fig = ckan.plot(beta=10, folder=f'./activations/{args.exp_name}{args.id}_pruned')
fig.savefig(output_dir / 'coxkan_pruned.png')
cindex_train = bootstrap_metric(ckan.cindex, df_train, N=100)['formatted']
if config['train_params']['early_stopping']: cindex_pruned_val = ckan.cindex(val)
else: cindex_pruned_val = None
cindex_pruned = bootstrap_metric(ckan.cindex, df_test, N=100)['formatted']

ckan_pru_str = f'CoxKAN (pruned) - train: {cindex_train}, test: {cindex_pruned}'; print(ckan_pru_str)
with open(output_dir / 'cindex.txt', 'a') as f:
    f.write(ckan_pru_str + '\n')
results['CoxKAN']['Pruned'] = {'train': cindex_train, 'test': cindex_pruned, 'val': cindex_pruned_val}

### Symbolic Fitting

# Initial symbolic fitting (categorical covariates and linear activations)
ckan.predict(pd.concat([df_train, df_test], axis=0))
for l in range(ckan.depth):
    for i in range(ckan.width[l]):
        for j in range(ckan.width[l+1]):
            if ckan.symbolic_fun[l].funs_name[j][i] != '0':
                if l == 0 and hasattr(ckan, 'categorical_covariates') and ckan.covariates[i] in ckan.categorical_covariates:
                    ckan.fix_symbolic(l,i,j,'categorical')
                else:
                    # try linear fit (accept if R2 > 0.95)
                    _, _, r2 = ckan.suggest_symbolic(l,i,j,lib=['x'], verbose=False)
                    if r2 > 0.95:
                        ckan.fix_symbolic(l,i,j,'x',verbose=False)
                        print(f'Fixed ({l},{i},{j}) as linear')

lib = list(SYMBOLIC_LIB.keys()).copy()

# Auto symbolic fitting for the remaining edges
if config['train_params']['early_stopping']: ckan.predict(pd.concat([val, df_test], axis=0))
else: ckan.predict(df_test)
if ckan.auto_symbolic(min_r2=0, lib=lib, verbose=True):
    cindex_symbolic = bootstrap_metric(ckan.cindex, df_test, N=100)['formatted']
    formula = ckan.symbolic_formula(floating_digit=4)[0][0]
    try:
        fig = ckan.plot(beta=20, folder=f'./activations/{args.exp_name}{args.id}_symbolic')
        fig.savefig(output_dir / 'coxkan_symbolic.png', dpi=600)
    except:
        print('Plotting coxkan_symbolic failed.')
        out = ckan.predict(df_test)
        print(f'Number of NaN predictions: {np.sum(np.isnan(out))}/{len(out)}')
else:
    # If symbolic fitting fails, set cindex to nan
    cindex_symbolic = np.nan
    formula = None
    print('Symbolic fitting failed.')

try:
    if config['train_params']['early_stopping']: cindex_symbolic_val = ckan.cindex(val)
    if cindex_symbolic_val/cindex_pruned_val < 0.9:
        warnings.warn("Symbolic fitting caused a reduction in the validation set C-Index by more than 10%, consider re-running the script with a different seed.", UserWarning)
except: pass 

### Save results
try: cindex_train = bootstrap_metric(ckan.cindex, df_train, N=100)['formatted']
except: cindex_train = np.nan

ckan_sym_str = f'CoxKAN (symbolic) - train: {cindex_train}, test: {cindex_symbolic}'; print(ckan_sym_str)
with open(output_dir / 'cindex.txt', 'a') as f:
    f.write(ckan_sym_str + '\n')

if 'sigmoid' in str(formula): formula = str(formula)

results['CoxKAN']['Symbolic'] = {'train': cindex_train, 'test': cindex_symbolic, 'formula': formula}
with open(output_dir / 'results.pkl', 'wb') as f:
    pickle.dump(results, f)

# save c-index results to txt file too
with open(output_dir / 'cindex.txt', 'w') as f:
    if run_coxph: f.write(coxph_str + '\n')
    if run_deepsurv: f.write(deepsurv_str + '\n')
    f.write(ckan_pre_str + '\n')
    f.write(ckan_pru_str + '\n')
    f.write(ckan_sym_str + '\n')