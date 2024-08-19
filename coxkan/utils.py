"""
CoxKAN Utility Functions
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import sympy
from lifelines.utils import concordance_index
import warnings
import scipy.stats as st
from kan.utils import fit_params, SYMBOLIC_LIB

# remove 'arcsin' from the symbolic library
del SYMBOLIC_LIB['arcsin']

def bootstrap_metric(metric_fn, df, N=100):
    """
    Bootstrap the confidence interval of a metric.

    Args:
    -----
        metric_fn : callable
            Metric function that takes a DataFrame as input.
        df : pd.DataFrame
            DataFrame to bootstrap.
        N : int
            Number of bootstrap samples. The default is 100.

    Returns:
    --------
        results : dict
            results['full'], metric of the full dataset.
            results['mean'], mean of the bootstrap samples.
            results['confidence_interval'], 95% confidence interval of the metric.
            results['formatted'], formatted string of the metric and confidence interval.
    """

    metrics = []
    size = len(df)

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace=True)
        df_ = df.iloc[resample_idx]
        df_ = df_.reset_index(drop=True)
        metric = metric_fn(df_)
        metrics.append(metric)
    
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))
    return {
        'full': metric_fn(df),
        'mean': mean,
        'confidence_interval': conf_interval,
        'formatted': f"{metric_fn(df):.6f} ({conf_interval[0]:.3f}, {conf_interval[1]:.3f})"
    }

class Logger:
    """
    Logger class to store training and testing metrics.
    """

    def __init__(self, early_stopping=False, stop_on='cindex'):
        """
        
        Args:
        -----
            early_stopping : bool
                Whether to use early stopping.
            stop_on : str
                Metric to use for early stopping. Either 'cindex' or 'loss'.
        """
        self.data = {}
        self.early_stopping = early_stopping
        self.stop_on = stop_on

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def plot(self):
        if not self.data:
            print("No data to plot.")
            return
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if 'train_loss' in self.data:
            ax[0].plot(self.data['train_loss'], label='train_loss')
            ax[0].set_title('Loss')
        if 'train_cindex' in self.data:
            ax[1].plot(self.data['train_cindex'], label='train_cindex')
            ax[1].set_title('C-Index')
        if 'val_loss' in self.data:
            ax[0].plot(self.data['val_loss'], label='val_loss')
        if 'val_cindex' in self.data:
            ax[1].plot(self.data['val_cindex'], label='val_cindex')

        # put vertical line at highest val_cindex
        if 'val_cindex' in self.data and self.early_stopping:
            if self.stop_on == 'cindex':
                best_epoch = np.argmax(self.data['val_cindex'])
            elif self.stop_on == 'loss':
                best_epoch = np.argmin(self.data['val_loss'])
            ax[0].axvline(best_epoch, color='k', linestyle='--', label='best_model')
            ax[1].axvline(best_epoch, color='k', linestyle='--', label='best_model')

        ax[0].grid(True); ax[1].grid(True)
        ax[0].legend(); ax[1].legend()

        return fig

def FastCoxLoss(log_h: Tensor, labels: Tensor, eps=1e-7) -> Tensor:
    """
    Simple and fast implementation of Cox Proportional Hazards Loss.
    Credit: https://github.com/havakv/pycox

    We just compute a cumulative sum. In the case of ties, this may not be the 
    exact true Risk set. This is a limitation, but fast.

    Args:
    -----
        log_h : Tensor
            Log-partial hazard.
        labels : Tensor
            Labels tensor: first column is time, second column is event indicator.
        eps : float
            Small value to prevent log(0).
    """
    
    # Sort by time
    durations, events = labels[:, 0], labels[:, 1]
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]

    # Compute the risk set
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)

    # Compute loss
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def set_seed(seed):
    """ Set seed for reproducibility. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    return seed

def add_symbolic(name, fn, sympy_fn):
    """
    Add a symbolic function to the symbolic library.

    Args:
    -----
        name : str
            Name of the symbolic function.
        fn : callable
            Function (lambda or torch)
        sympy_fn : callable
            Sympy function

    Returns:
    --------
        None
    """

    globals()[name] = sympy_fn
    SYMBOLIC_LIB[name] = (fn, globals()[name])

def count_parameters(model):
    """ Count the number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_fun(inputs, outputs, category_map):
    """
    Create a categorical (discrete) function. 
    
    Primary purpose is for creating symbolic activation functions for categorical covariates.
    The function accepts an array of inputs and outputs of a given (non-symbolic) activation function,
    as well as a dictionary mapping the encoded values of a categorical variable to the original category name.
    It returns a function that maps the inputs to the outputs (exactly as the activation function did), as well
    as a discrete Sympy function that represents the categorical mapping.

    Args:
    -----
        inputs : torch.Tensor
            Inputs to the function.
        outputs : torch.Tensor
            Outputs of the function.
        category_map : dict
            Dictionary mapping encoded values of a category to the original category name.

    Returns:
    --------
        func : callable
            Function that maps inputs to outputs based on the categorical mapping.
        sympy_func : callable
            Discrete Sympy function representing the categorical mapping.
    """

    # Check that all inputs are in the category map
    unique_inputs = [round(i.item(), 3) for i in inputs.unique()]
    for inpt in unique_inputs:
        assert inpt in category_map.keys()

    # Create a mapping from input to output
    mapping = {}
    for idx, x in enumerate(inputs):
        x = round(x.item(), 3)
        if x not in mapping:
            mapping[x] = outputs[idx].item()
        else:
            assert round(mapping[x],6) == round(outputs[idx].item(), 6)
    
    # Create the function that maps inputs to outputs
    def func(x):
        shape = x.shape
        x = x.flatten()
        try:
            return torch.tensor([mapping[round(x_i.item(), 3)] for x_i in x]).reshape(shape)
        except:
            print(f'\n\n\n')
            print(mapping)
            raise ValueError(f"Input not found in categorical mapping.")
    
    # Create the discrete Sympy function
    def sympy_func(x):
        conditions = []
        for i in unique_inputs:
            out = mapping[i]
            value = category_map[i] if isinstance(category_map[i], float) else sympy.symbols(str(category_map[i]))
            conditions.append((out, sympy.Eq(x, value)))
        return sympy.Piecewise(*conditions, (sympy.nan, True), evaluate=False)

    return func, sympy_func