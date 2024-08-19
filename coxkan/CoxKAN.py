"""
Main module for CoxKAN class.
"""

import torch
from kan import KAN
from kan.LBFGS import LBFGS
from lifelines.utils import concordance_index
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
from kan.utils import fit_params
import os
from tqdm import tqdm
from pathlib import Path
import uuid
from torch import Tensor

from .utils import FastCoxLoss, categorical_fun, Logger, SYMBOLIC_LIB

TEMP_CKPT_DIR = Path(__file__).parent / '_ckpt'
os.makedirs(TEMP_CKPT_DIR, exist_ok=True)

class CoxKAN(KAN):
    """
    CoxKAN class

    Attributes:
    ------------
        act_fun: a list of KANLayer
            KANLayers
        depth: int
            depth of KAN
        width: list
            number of neurons in each layer. e.g., [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
        grid: int
            the number of grid intervals
        k: int
            the order of piecewise polynomial
        base_fun: fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun: a list of Symbolic_KANLayer
            Symbolic_KANLayers

    Methods:
    --------
        __init__():
            initalize a CoxKAN model
        process_data():
            preprocess dataset and register metadata
        train():
            train the model
        cindex():
            compute concordance index
        predict():
            predict the log-partial hazard
        predict_partial_hazard():
            predict the partial hazard (exp of log-partial hazard)
        prune_edges():
            prune edges (activation functions) of the model
        prune_nodes():
            prune nodes (neurons) of the model
        fix_symbolic():
            set (l,i,j) activation to be symbolic (specified by fun_name)
        plot():
            plot the model
        plot_act():
            plot a specific activation function
        suggest_symbolic():
            find the best symbolic function for a specific activation (highest r2)
        auto_symbolic():
            automatic symbolic fitting
        symbolic_formula():
            obtain the symbolic formula of the full model
        symbolic_rank_terms():
            calculate standard devation of each term in symbolic formula
    """

    def __init__(self, **kwargs):
        '''
        Initalize a CoxKAN model
        
        Keyword Args:
        -----
            width : list of int
                :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            noise_scale : float
                initial injected noise to spline. Default: 0.1.
            base_fun : fun
                the residual function b(x). Default: torch.nn.SiLU().
            symbolic_enabled : bool
                compute or skip symbolic computations (for efficiency). By default: True. 
            bias_trainable : bool
                bias parameters are updated or not. By default: True
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            
        Returns:
        --------
            self
        '''
        if kwargs.get('base_fun')=='silu':
            kwargs['base_fun'] = torch.nn.SiLU()
        elif kwargs.get('base_fun')=='linear':
            kwargs['base_fun'] = torch.nn.Identity()
        super(CoxKAN, self).__init__(**kwargs)

    def process_data(self, df_train, df_test, duration_col, event_col, covariates=None, categorical_covariates=True, normalization='minmax'):
        """
        Preprocess dataset and register metadata via the following steps:
            - Encode categorical covariates via label-encoding (if categorical_covariates is not None)
            - Normalize covariates
            - Register metadata: duration_col, event_col, covariates, normalizer, categorical_covariates and category_maps (maps from the encoded values of each category to the original names)
        
        Args:
        -----
            df_train : pd.DataFrame
                training dataset
            df_test : pd.DataFrame
                testing dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event
            covariates : list
                list of covariates. If None, all columns except duration_col and event_col are used.
            categorical_covariates : bool or list
                If True, categorical covariates are automatically detected and label encoded. 
                If a list is provided, only the covariates in the list are label encoded.
            normalization : str
                normalization method: 'minmax' for :math:`(x - min(x))/(max(x) - min(x))`, 'standard' for :math:`(x - mean(x))/std(x)`, or 'none'

        Returns:
        --------
            df_train : pd.DataFrame
                training dataset with processed covariates
            df_test : pd.DataFrame
                testing dataset with processed covariates
        """

        if covariates is None: # if covariates are not provided, use all columns except duration_col and event_col
            covariates = df_train.columns.drop([duration_col, event_col])

        # check for cases where there is just one value
        for col in covariates:
            if len(df_train[col].unique()) == 1:
                raise ValueError(f"Column {col} has only one unique value. Please remove it from covariates.")

        # register metadata
        self.duration_col, self.event_col, self.covariates = duration_col, event_col, covariates

        X = pd.concat([df_train[covariates], df_test[covariates]])

        # find categorical covariates (type is 'category', or has less than 5 unique values)
        if categorical_covariates == True:
            categorical_covariates = []
            for col in covariates:
                if len(X[col].unique()) < 5:
                    categorical_covariates.append(col)
                elif X[col].dtype.name == 'category':
                    categorical_covariates.append(col)

        # encode categorical covariates via label-encoding
        if categorical_covariates:
            category_maps = {}
            for cat in categorical_covariates:
                category_maps[cat] = dict(enumerate(X[cat].astype('category').cat.categories))
                X[cat] = X[cat].astype('category').cat.codes
                X[cat] = X[cat].astype('float32')

        df_train[covariates] = X[:len(df_train)]
        df_test[covariates] = X[len(df_train):]

        if normalization is None or normalization == 'none':
            return df_train, df_test
        
        # detect high collinearity
        corr = pd.concat([df_train, df_test]).corr()
        np.fill_diagonal(corr.values, 0)
        if (np.abs(corr) > 0.999999).sum().sum() > 0:
            print("Warning: High collinearity detected. Consider removing one of the highly correlated features.")

        # normalize covariates
        normalizer = []
        if normalization == 'minmax':
            normalizer.append(X.min())
            normalizer.append(X.max() - X.min())
        elif normalization == 'standard':
            normalizer.append(X.mean())
            normalizer.append(X.std())
        else:
            raise NotImplementedError("Normalization can be 'minmax', 'standard' or 'none'.")

        df_train[covariates] = (df_train[covariates] - normalizer[0]) / normalizer[1]
        df_test[covariates] = (df_test[covariates] - normalizer[0]) / normalizer[1]

        # convert the keys of each of the category maps to be their normalized values
        if categorical_covariates:
            for cat in category_maps.keys():
                items = list(category_maps[cat].items())
                for key, val in items:
                    # remove the old key
                    category_maps[cat].pop(key)
                    # add the new key
                    new_key = round((key - normalizer[0][cat]) / normalizer[1][cat], 3)
                    category_maps[cat][new_key] = val
        
            # register
            self.categorical_covariates = categorical_covariates
            self.category_maps = category_maps

        # register normalizer
        self.normalizer = normalizer

        return df_train, df_test

    def train(self, df_train, df_val=None, duration_col='duration', event_col='event', covariates=None, 
              opt="Adam", lr=0.01, steps=100, batch=-1, early_stopping=False, stop_on='cindex',
              log=1, lamb=0., lamb_l1=1., lamb_entropy=0., 
              lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, stop_grid_update_step=50, 
              small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, 
              in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu', progress_bar=True):
        """
        Train the model.

        Args:
        -----
            df_train : pd.DataFrame
                training dataset
            df_val : pd.DataFrame
                validation dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event
            covariates : list
                list of covariates. If None, all columns except duration_col and event_col are used.
            opt : str
                optimizer. 'Adam' or 'LBFGS'
            lr : float
                learning rate
            steps : int
                number of steps
            batch : int
                batch size. If -1, use all samples.
            log : int
                log frequency
            lamb : float
                overall regularization strength
            lamb_l1 : float
                l1 regularization strength
            lamb_entropy : float
                entropy regularization strength
            lamb_coef : float
                spline coefficient regularization strength
            lamb_coefdiff : float
                spline coefficient difference regularization strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            metrics : list
                additional metrics to log
            sglr_avoid : bool
                avoid nan in SGLR
            save_fig : bool
                save figures
            beta : float
                beta for plotting
            save_fig_freq : int
                save figure frequency
            img_folder : str
                folder to save figures
            device : str
                device to use (no need to change as gpu is typically slower)

        Returns:
        --------
            log : dict
                log['train_loss'], 1D array of training losses (Cox loss)
                log['val_loss'], 1D array of val losses (Cox loss)
                log['train_cindex'], 1D array of training concordance index
                log['val_cindex'], 1D array of val concordance index
                log['reg'], 1D array of regularization (regularization in the total loss)
        """
        
        # spline grid update frequency
        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        ### Register metadata
        if covariates is None: # if covariates are not provided, use all columns except duration_col and event_col
            covariates = df_train.columns.drop([duration_col, event_col])

        self.duration_col, self.event_col, self.covariates = duration_col, event_col, covariates

        ### Prepare data
        X_train = torch.tensor(df_train[covariates].values, dtype=torch.float32)
        y_train = torch.tensor(df_train[[duration_col, event_col]].values, dtype=torch.float32)
        if df_val is not None:
            X_val = torch.tensor(df_val[covariates].values, dtype=torch.float32)
            y_val = torch.tensor(df_val[[duration_col, event_col]].values, dtype=torch.float32)

        ### Define regularization
        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_
        
        ### Define optimizer
        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        ### Init log
        logger = Logger(early_stopping=early_stopping, stop_on=stop_on)
        logger['train_loss'], logger['val_loss'], logger['train_cindex'], logger['val_cindex'], logger['reg'] = [], [], [], [], []
        if metrics != None:
            for i in range(len(metrics)):
                logger[metrics[i].__name__] = []

        ### Define batch size
        if batch == -1 or batch > X_train.shape[0]:
            batch_size = X_train.shape[0]
        else:
            batch_size = batch

        ### Define closure (inner function for optimizer.step)
        global train_loss, reg_
        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(X_train[train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = FastCoxLoss(pred[id_], y_train[train_id][id_].to(device))
            else:
                train_loss = FastCoxLoss(pred, y_train[train_id].to(device))
            reg_ = reg(self.acts_scale)
            loss = train_loss + lamb * reg_
            loss.backward()
            return loss
        
        ### Generate best model hash for early stopping
        if early_stopping:
            best_model_hash = uuid.uuid4().hex
        
        if save_fig:
            os.makedirs(img_folder, exist_ok=True)

        ### Train
        if progress_bar: pbar = tqdm(range(steps), desc='description', ncols=100)
        else: pbar = range(steps)
        best_cindex = 0
        best_val_loss = np.inf
        for step, _ in enumerate(pbar):

            # Sample batch (typically, we use all samples for training)
            train_id = np.random.choice(X_train.shape[0], batch_size, replace=False)

            # Update spline grids
            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(X_train[train_id].to(device))

            # Update
            optimizer.step(closure)

            if metrics != None:
                for i in range(len(metrics)):
                    log[metrics[i].__name__].append(metrics[i]().item())

            logger['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            logger['train_cindex'].append(self.cindex(df_train))
            logger['reg'].append(reg_.cpu().detach().numpy())
            if df_val is not None:
                val_loss = FastCoxLoss(self.forward(X_val.to(device)), y_val.to(device))
                val_loss = torch.sqrt(val_loss).cpu().detach().numpy()
                cindex_val = self.cindex(df_val)
                if early_stopping and step > 1:
                    if stop_on == 'cindex' and cindex_val > best_cindex:
                        best_cindex = cindex_val
                        self.save_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
                    elif stop_on == 'loss' and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
                logger['val_loss'].append(val_loss)
                logger['val_cindex'].append(cindex_val)

            if _ % log == 0:
                if df_val is not None: pbar_desc = f"train loss: {logger['train_loss'][-1]:.2e} | val loss: {logger['val_loss'][-1]:.2e}"
                else: pbar_desc = f"train loss: {logger['train_loss'][-1]:.2e}"
                if progress_bar: pbar.set_description(pbar_desc)

            if save_fig and _ % save_fig_freq == 0:
                if in_vars is None: in_vars = list(covariates)
                if out_vars is None: out_vars = [r'$\hat{\theta}$']
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

        if early_stopping:
            self.load_ckpt(TEMP_CKPT_DIR / f'{best_model_hash}.pt', verbose=False)
            print('Best model loaded (early stopping).')
            os.remove(TEMP_CKPT_DIR / f'{best_model_hash}.pt')
            _ = self.predict(df_val) # necessary forward pass 

        return logger

    def cindex(self, df, duration_col=None, event_col=None):
        """
        Compute model's concordance index on a dataset.

        Args:
        -----
            df : pd.DataFrame
                dataset
            duration_col : str
                column name for duration
            event_col : str
                column name for event

        Returns:
        --------
            cindex : float
                concordance index
        """

        # if duration_col and event_col are not provided, use the registered metadata
        if duration_col is None and event_col is None:
            assert hasattr(self, 'duration_col') and hasattr(self, 'event_col'), "Dataset metadata not registered. Please train model or use process_data."
            duration_col, event_col = self.duration_col, self.event_col

        # compute concordance index
        X = torch.tensor(df.drop([self.duration_col, self.event_col], axis=1).values, dtype=torch.float32)
        log_ph = self(X).detach().numpy().flatten()

        return concordance_index(df[self.duration_col], -log_ph, df[self.event_col])

    def predict(self, df):
        """
        Predict log-partial hazard for all samples in a dataset.

        Args:
        -----
            df : pd.DataFrame
                dataset

        Returns:
        --------
            log_ph : pd.Series
                log-partial hazard
        """
    
        assert hasattr(self, 'duration_col') and hasattr(self, 'event_col') and hasattr(self, 'covariates'), "Dataset metadata not registered. Please train model or use process_data."
        X = torch.tensor(df[self.covariates].values, dtype=torch.float32)
        return pd.Series(self(X).cpu().detach().numpy().flatten(), index=df.index)

    def predict_partial_hazard(self, df):
        """
        Predict partial hazard for all samples in a dataset (exp of log-partial hazard).

        Args:
        -----
            df : pd.DataFrame
                dataset

        Returns:
        --------
            partial_hazard : pd.Series
                partial hazard
        """
        return np.exp(self.predict(df))
    
    def prune_edges(self, threshold=0.02, verbose=True):
        """
        Prune edges (activation functions) of the model based on a threshold of the L1 norm
        of that activation.

        Args:
        -----
            threshold : float
                any activation with L1 norm less than this threshold will be pruned
            verbose : bool
                If True, print pruned activations

        Returns:
        --------
            None
        """
        # loop through all activations
        for l in range(self.depth):
            for i in range(self.width[l]):
                for j in range(self.width[l+1]):
                    if self.acts_scale[l][j][i] < threshold:
                        super(CoxKAN, self).remove_edge(l, i, j)        # remove edge
                        self.fix_symbolic(l, i, j, '0', verbose=False)  # set symbolic activation to 0
                        self.acts_scale[l][j][i] = 0                    # set scale to 0

                        if verbose: print(f'Pruned activation ({l},{i},{j})')
                        assert self.symbolic_fun[l].funs_name[j][i] == '0'
                        assert self.symbolic_fun[l].mask[j][i] == 1
                        assert self.act_fun[l].mask[j * self.width[l] + i] == 0
                        assert self.acts_scale[l][j][i] == 0

    def prune_nodes(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        '''
        Prune nodes (neurons) of the model based on a threshold of the L1 norm of the incoming
        and outgoing activations of that neuron. This method is just slightly adapted from
        the original KAN.prune().
        
        Args:
        -----
            threshold : float
                any neuron which has all incoming and outgoing activations with L1 norm less than this threshold will be pruned
            mode : str
                "auto" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "manual", active_neuron_id is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists
                For example, [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.
            
        Returns:
        --------
            model2 : CoxKAN
                pruned model
        '''
        mask = [torch.ones(self.width[0], )]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(torch.where(overall_important == True)[0])
        active_neurons.append(list(range(self.width[-1])))
        mask.append(torch.ones(self.width[-1], ))

        self.mask = mask  # this is neuron mask for the whole model

        # update act_fun[l].mask
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = CoxKAN(width=copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun, device='cpu')
        model2.load_state_dict(self.state_dict())

        # copy other attributes
        dic = {}
        for k, v in self.__dict__.items():
            if k[0] != '_':
                setattr(model2, k, v)

        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                model2.biases[i].weight.data = model2.biases[i].weight.data[:, active_neurons[i + 1]]

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])
            model2.width[i] = len(active_neurons[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])

        return model2
    
    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True, random=False):
        '''
        Set (l,i,j) activation to be symbolic (specified by fun_name).
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]
        
        Returns:
        --------
            None or r2 (coefficient of determination)
  
        '''
        self.set_mode(l, i, j, mode="s")
        if fun_name == 'categorical':
            assert l == 0, "Only input layer can have categorical activations"
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            category_map = self.category_maps[self.covariates[i]]
            fun, fun_sympy = categorical_fun(inputs=x, outputs=y, category_map=category_map)
            self.symbolic_fun[l].funs_sympy[j][i] = fun_sympy
            self.symbolic_fun[l].funs_name[j][i] = fun_name
            self.symbolic_fun[l].funs[j][i] = fun 
            self.symbolic_fun[l].affine.data[j][i] = torch.tensor([1.,0.,1.,0.])
            return None
        elif not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            return None
        else:
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)

            # if in output layer, fix output bias to zero
            if l == len(self.width) - 2:
                self.symbolic_fun[l].affine.data[j][i][3] = 0.

            return r2

    def plot(self, show_vars=False, **kwargs):
        """ Plot the model. 
        
        Args:
        -----
            show_vars : bool
                If True, show the registered covariates on the plot. Default: False
            **kwargs : Keyword arguments to be passed to KAN.plot()
        
        Keyword Args:   
        -------------
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title

        Returns:
        --------
            fig : Figure
                the figure
        """

        # re-apply mask
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if  self.symbolic_fun[l].funs_name[j][i] == '0' and self.symbolic_fun[l].mask[j, i] > 0.:
                        self.acts_scale[l][j][i] = 0.

        if show_vars:
            super(CoxKAN, self).plot(in_vars=list(self.covariates), out_vars=[r'$\hat{\theta}(\mathbf{x})$'], **kwargs)
        else:
            super(CoxKAN, self).plot(**kwargs)
        return plt.gcf()

    def plot_act(self, l, i, j):
        """
        Plot activation function phi_(l,i,j)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
        """
        # obtain inputs (pre-activations) and outputs (post-activations)
        inputs = self.spline_preacts[l][:,j,i]
        outputs = self.spline_postacts[l][:,j,i]

        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]

        fig = plt.figure()
        plt.plot(inputs, outputs, marker="o")
        return fig

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True):
        ''' 
        Suggest the symbolic candidates of activation function phi_(l,i,j)
        
        Args:
        -----
            l : int
                layer index
            i : int 
                input neuron index
            j : int 
                output neuron index
            lib : dic
                library of symbolic bases. If lib = None, the global default library will be used. 
            topk : int
                display the top k symbolic functions (according to r2)
            verbose : bool
                If True, more information will be printed.
           
        Returns:
        --------
            fun_name : str
                suggested symbolic function name
            fun : fun
                suggested symbolic function
            r2 : float
                coefficient of determination of best suggestion
            
        '''
        if hasattr(self, 'categorical_covariates') and l == 0:
            if self.covariates[i] in self.categorical_covariates and self.symbolic_fun[l].funs_name[j][i] != '0':
                return 'categorical', None, 1
            
        r2s = []

        if lib == None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = SYMBOLIC_LIB[item]

        lib_attempted = []
        for (name, fn) in symbolic_lib.items():
            try:
                r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False)
                r2s.append(r2.item())
                lib_attempted.append((name, fn))
            except Exception as e:
                if verbose:
                    print(f'Error in fitting "{name}": {e}')

        self.unfix_symbolic(l, i, j)

        sorted_ids = np.argsort(r2s)[::-1][:topk]
        r2s = np.array(r2s)[sorted_ids][:topk]
        topk = np.minimum(topk, len(lib_attempted))
        if verbose == True:
            print('function', ',', 'r2')
            for i in range(topk):
                print(list(lib_attempted)[sorted_ids[i]][0], ',', r2s[i])

        best_name = list(lib_attempted)[sorted_ids[0]][0]
        best_fn = list(lib_attempted)[sorted_ids[0]][1]
        best_r2 = r2s[0]
        return best_name, best_fn, best_r2
    
    def plot_best_suggestion(l, i, j, lib=None, a_range=(-10,10), b_range=(-10,10), verbose=1):
        """
        Plot the best symbolic suggestion for activation function phi_(l,i,j)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            lib : None or a list of function names
                the symbolic library 
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : int
                verbosity

        Returns:
        --------
            fig : Figure
                the figure
        """
        x = self.spline_preacts[l][:,j,i]
        y = self.spline_postacts[l][:,j,i]

        # they are not ordered yet
        rank = np.argsort(x)
        x = x[rank]
        y = y[rank]

        fn_name, _, r2 = self.suggest_symbolic(l, i, j, lib=lib, a_range=a_range, b_range=b_range, verbose=verbose)

        # minimise |y-(cf(ax+b)+d)|^2 w.r.t a,b,c,d
        func = SYMBOLIC_LIB[fn_name][0]
        (a, b, c, d), r2 = fit_params(x, y, func, a_range=a_range, b_range=b_range, verbose=verbose)

        y_pred = c*func(a*x+b)+d

        fig, ax = plt.subplots()

        ax.scatter(x, y, label="Activation")
        ax.plot(x, y_pred, color='red', linestyle='--', label=f"Symbolic Fit")
        ax.set_title(f"{c:.3f}{fn_name}({a:.3f}x + {b:.3f}) + {d:.3f}")
        ax.legend()
        return fig

    def auto_symbolic(self, min_r2=0, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1):
        '''
        Automatic symbolic regression: using best suggestion from suggest_symbolic to replace activations with symbolic functions.
        This method is just slightly adapted from the original KAN.auto_symbolic().
        
        Args:
        -----
            min_r2 : float
                minimum r2 to accept the symbolic formula
            lib : None or a list of function names
                the symbolic library 
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : int
                verbosity
                
        Returns:
        --------
            bool: True if all activations are successfully replaced by symbolic functions, False otherwise
        '''

        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0.:
                        if verbose:
                            print(f'skipping ({l},{i},{j}) since already symbolic')
                    else:
                        name, fn, r2 = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=verbose > 1)
                        if r2 >= min_r2:
                            self.fix_symbolic(l, i, j, name, verbose=verbose > 1)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={r2}')
                        else:
                            print(f'No symbolic formula found for ({l},{i},{j})')
                            return False 
                        
        return True


    def symbolic_formula(self, floating_digit=None, var=None, normalizer=None, simplify=False, output_normalizer = None ):
        '''
        Obtain the symbolic formula.
        
        Args:
        -----
            floating_digit : int
                the number of digits to display
            var : list of str
                the name of variables (if not provided, by default using ['x_1', 'x_2', ...])
            normalizer : [mean array (floats), varaince array (floats)]
                the normalization applied to inputs
            simplify : bool
                If True, simplify the equation at each step (usually quite slow), so set up False by default.
            output_normalizer: [mean array (floats), varaince array (floats)]
                the normalization applied to outputs
            
        Returns:
        --------
            symbolic formula : sympy function
                the symbolic formula
            x0 : list of sympy symbols
                the list of input variables
        
        '''
        symbolic_acts = []
        x = []

        def ex_round(ex1, floating_digit=floating_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2
        
        if normalizer is None and hasattr(self, 'normalizer'):
            normalizer = self.normalizer

        # define variables
        if var is None:
            if hasattr(self, 'covariates'):
                x = [sympy.symbols(var_.replace(' ', '_')) for var_ in self.covariates]
            else:
                for ii in range(1, self.width[0] + 1):
                    exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                    exec(f"x.append(x{ii})")
        else:
            x = [sympy.symbols(var_.replace(' ', '_')) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = np.array(normalizer[0])
            std = np.array(normalizer[1])
            if hasattr(self, 'categorical_covariates'):
                for i, var_ in enumerate(self.covariates):
                    if var_ not in self.categorical_covariates:
                        x[i] = (x[i] - mean[i]) / std[i]
            else:
                x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width) - 1):
            y = []
            for j in range(self.width[l + 1]):
                yj = 0.
                for i in range(self.width[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    if l == len(self.width) - 2: d = 0
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    fun_name = self.symbolic_fun[l].funs_name[j][i]
                    try:
                        if fun_name == 'categorical':
                            assert a == 1 and b == 0 and c == 1 and d == 0
                            yj += sympy_fun(x[i])
                        else:
                            yj += c * sympy_fun(a * x[i] + b) + d
                    except Exception as e:
                        print('Error: ', e)
                if simplify == True:
                    y.append(sympy.simplify(yj + self.biases[l].weight.data[0, j]))
                else:
                    if l == len(self.width) - 2:
                        y.append(yj)
                    else:
                        y.append(yj + self.biases[l].weight.data[0, j])

            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer

        if floating_digit is None:
            return symbolic_acts[-1], x0

        self.symbolic_acts = [[ex_round(symbolic_acts[l][i]) for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]

        out_dim = len(symbolic_acts[-1])
        return [ex_round(symbolic_acts[-1][i]) for i in range(len(symbolic_acts[-1]))], x0

    def save_ckpt(self, save_path='ckpt.pt', verbose=True):
        ''' Save the current model as checkpoint '''

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        for l in range(self.depth):
            if 1 in self.symbolic_fun[l].mask:
                raise NotImplementedError('Saving of pruned or symbolic models not supported yet.')

        state = {
            'state_dict': self.state_dict(),
        }
        for k, v in self.__dict__.items():
            if k[0] != '_':
                state[k] = v

        torch.save(state, save_path)
        if verbose: print(f'Saved model to {save_path}')

    def load_ckpt(self, ckpt_path, verbose=True):
        ''' Load model from checkpoint '''
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])
        for k, v in state.items():
            if k != 'state_dict':
                setattr(self, k, v)
        if verbose: print(f'Loaded model from {ckpt_path}')

    def symbolic_rank_terms(self, floating_digit=5, z_score_threshold=5, normalizer=None):
        """
        Calculate the standard deviation of each term in the symbolic expression of CoxKAN.

        Standard deviation can be used as a measure of importance of each term in the symbolic expression.
        The terms with higher standard deviation are more important. A caveat here is that terms with
        outliers in their outputs may have higher standard deviation, which may not necessarily mean they
        are more important. To address this, we remove outliers iteratively based on Z-score until no
        outliers are left.

        Args:
        -----
            floating_digit : int
                the number of digits to display
            z_score_threshold : int
                the threshold of Z-score for removing outliers
            normalizer : [mean array (floats), varaince array (floats)]
                the normalization applied to inputs

        Returns:
        --------
            terms_std : dict
                dictionary of terms and their standard deviations
        """

        for l in range(self.depth):
            if not (self.symbolic_fun[l].mask == 1).all():
                raise ValueError('All activation functions must be symbolic for ranking.')
            
        def zscore(arr):
            return (arr - np.mean(arr)) / np.std(arr)
            
        def remove_outliers(arr):
            """ 
            Remove outliers from an array based on Z-score iteratively until no outliers are left.
            """
            z_scores = np.abs(zscore(arr))
            while np.any(z_scores > z_score_threshold):
                arr = arr[np.abs(zscore(arr)) < z_score_threshold]
                z_scores = np.abs(zscore(arr))
            return arr

        def ex_round(ex1):
            """
            Round the floating point numbers in a sympy expression.
            """
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, floating_digit))
            return ex2

        if normalizer is None and hasattr(self, 'normalizer'):
            normalizer = self.normalizer

        with torch.no_grad():
            ### Get the terms in the KAN symbolic expression and their standard deviations
            terms_std = {}

            if len(self.width) == 2:
                for i in range(self.width[0]):
                    l = 0; j = 0
                    fun_name = self.symbolic_fun[l].funs_name[j][i]
                    fun = self.symbolic_fun[l].funs[j][i]
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]

                    if fun_name != '0':
                        outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                        outputs = remove_outliers(outputs)
                        x = self.covariates[i]
                        x = x.replace(' ', '_')
                        if normalizer is not None:
                            x = (sympy.symbols(x) - normalizer[0][i]) / normalizer[1][i]
                        else: 
                            x = sympy.symbols(x)
                        sympy_trans = self.symbolic_fun[l].funs_sympy[j][i]
                        term = c * sympy_trans(a * x + b)
                        term = ex_round(term)
                        term = f'({l},{i},{j}) ' + str(term)
                        terms_std[term] = outputs.std()
                        
            elif len(self.width) == 3:
                for i in range(self.width[1]):
                    l = 1
                    j = 0
                    fun_name = self.symbolic_fun[l].funs_name[j][i]
                    fun = self.symbolic_fun[l].funs[j][i]
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    if fun_name != '0':
                        # if the final layer activation is non-linear, it is an interaction term
                        if fun_name != 'x':
                            outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                            outputs = remove_outliers(outputs)
                            fun_name = f'({l},{i},{j}) {fun_name} interaction term'
                            terms_std[fun_name] = outputs.std()
                        # if the final layer activation is linear, it contains many isolation terms
                        else:
                            assert fun(1) == 1, f'Function {fun} is not linear'
                            j = i # current node becomes the output node
                            l = 0 # input layer
                            for i in range(self.width[l]):
                                if self.symbolic_fun[l].funs_name[j][i] != '0':

                                    # calculate standard deviation of the term (excluding outliers)
                                    outputs = self.spline_postacts[l][:, j, i].detach().numpy().flatten()
                                    outputs = c * (a * outputs + b) + d
                                    outputs = remove_outliers(outputs)
                                    std = outputs.std().item()

                                    # get symbolic expression of the term
                                    x = self.covariates[i]
                                    x = x.replace(' ', '_')
                                    if normalizer is not None:
                                        x = (sympy.symbols(x) - normalizer[0][i]) / normalizer[1][i]
                                    else: 
                                        x = sympy.symbols(x)
                                    a_, b_, c_, d_ = self.symbolic_fun[l].affine[j, i]
                                    sympy_trans = self.symbolic_fun[l].funs_sympy[j][i]
                                    transformation = c_ * sympy_trans(a_ * x + b_)
                                    term = c * a * transformation
                                    term = ex_round(term)
                                    term = f'({l},{i},{j}) ' + str(term)

                                    terms_std[term] = std
            else:
                raise NotImplementedError('Ranking terms is currently only supported for models with up to 1 hidden layer.')
        return terms_std
