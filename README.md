<h1 align="center">
    <img src="media/coxkan_logo.png" alt="CoxKAN" width="500">
</h1>

# CoxKAN: Kolmogorov-Arnold Networks for Survival Analysis

<p align="center">
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#datasets">Datasets</a> •
    <a href="#reproducibility">Reproducibility</a> •
    <a href="#credits">Credits</a>
</p>

This repository contains the codes accompanying the paper "CoxKAN: Kolmogorov-Arnold Networks for Interpretable, High-Performance Survival Analysis".
- Paper: [ArXiv](https://arxiv.org/abs/2409.04290).
- Installation: `pip install coxkan`
- Documentation: [Read-the-Docs](https://coxkan.readthedocs.io/en/latest/).
- Quick-start: `tutorials/intro.ipynb`

**Repo Structure:**
```
├── checkpoints/        # Results / checkpoints from paper
├── configs/            # Model configuration files
├── coxkan/             # CoxKAN package 
├── data/               # Data 
├── docs/               # Documentation
├── media/              # Figures used in paper
├── reprod/             # Reproducability instructions/code
├── tutorials/          # Tutorials for CoxKAN
|
# standard stuff:
├── .gitignore         
├── LICENSE          
├── README.md          
└── setup.py            
```

# Installation 

### Pip
CoxKAN can be installed via:

```bash
pip install coxkan
```

### Git

Alternatively, may desire the full codebase and environment that was used to produce all results in the associated paper:

```bash
git clone https://github.com/knottwill/CoxKAN.git 
cd CoxKAN
pip install -r reprod/requirements.txt
```

Please refer to reproducibility instructions in `reprod/README.md`.

# Usage

Find tutorials in `tutorials/` or [Read-the-Docs](https://coxkan.readthedocs.io/en/latest/)

### Example
```python
from coxkan import CoxKAN
from coxkan.datasets import metabric 

df_train, df_test = metabric.load(split=True)

ckan = CoxKAN(width=[len(metabric.covariates), 1])

_ = ckan.train(
    df_train, 
    df_test, 
    duration_col='duration', 
    event_col='event',
    steps=100)

# evaluate model
ckan.cindex(df_test)
>>> 0.6441975461899737
```

### CoxKAN Package

The `coxkan/` package has 4 main components: 
```
coxkan/
    ├── datasets/             # datasets subpackage
    ├── CoxKAN.py             # CoxKAN model
    ├── utils.py              # utility functions
    └── hyperparam_search.py  # hyperparameter searching
```


# Datasets

### Synthetic Datasets

`coxkan.datasets.create_dataset` makes it easy to generate synthetic survival data assuming a proportional-hazards, time-independant hazard function: $$h = h_0 e^{\theta(\mathbf{x})} \rightarrow T_s \sim \text{Exp}(h)$$ and uniform censoring distribution $T_c \sim \text{Uniform}(0, T_{max})$.

In the example below, we use a log-partial hazard of $\theta(\mathbf{x}) = 5 e^{-2(x_1^2 + x_2^2)}$ and a baseline hazard of $h_0 = 0.01$. 

```python
from coxkan.datasets import create_dataset

log_partial_hazard = lambda x1, x2: 5*np.exp(-2*(x1**2 + x2**2))
df = create_dataset(log_partial_hazard, baseline_hazard=0.01, n_samples=10000)
```

### Clinical Datasets

5 clinical datasets are available with the `coxkan.datasets` subpackage (inspired by [pycox](https://github.com/havakv/pycox)). For example:

```python
from coxkan.datasets import gbsg
df_train, df_test = gbsg.load(split=True)
```

You can decide where to store them using the `COXKAN_DATA_DIR` environment variable.

<table>
    <tr>
        <th>Dataset</th>
        <th>Description</th>
        <th>Source</th>
    </tr>
    <tr>
        <td>GBSG</td>
        <td>
        The Rotterdam & German Breast Cancer Study Group.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">DeepSurv</a>
    </tr>
    <tr>
        <td>METABRIC</td>
        <td>
        The Molecular Taxonomy of Breast Cancer International Consortium.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">DeepSurv</a>
    </tr>
    <tr>
        <td>SUPPORT</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">DeepSurv</a>
    </tr>
    <tr>
        <td>NWTCO</td>
        <td>
        National Wilm's Tumor Study.
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">Rdatasets</a>
    </tr>
    <tr>
        <td>FLCHAIN</td>
        <td>
        Assay of Serum Free Light Chain.
        </td>
        <td><a href="https://github.com/vincentarelbundock/Rdatasets">Rdatasets</a>
    </tr>
</table>

Unfortunately, [DeepSurv](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data) did not retain the column names. We manually restored the names by obtaining the datasets elsewhere and comparing the columns (then we can use the same train/test split):
- GBSG: https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman
- SUPPORT: https://hbiostat.org/data/repo/support2csv.zip
- METABRIC: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric

### Genomics Datasets

We curated 4 genomics datasets from The Cancer Genome Atlas Program (TCGA). The raw or pre-processed data is available by request - please email me at knottenbeltwill@gmail.com. 

Two of the datasets (GBMLGG, KIRC) were the unaltered datasets used in [Pathomic Fusion](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10339462/)

<table>
    <tr>
        <th>Dataset</th>
        <th>Description</th>
        <th>Source</th>
    </tr>
    <tr>
        <td>STAD</td>
        <td>
        Stomach Adenocarcinoma.
        </td>
        <td><a href="https://www.cancer.gov/ccg/research/genome-sequencing/tcga">TCGA</a>
    </tr>
    <tr>
        <td>BRCA</td>
        <td>
        Breast Invasive Carcinoma.
        </td>
        <td><a href="https://www.cancer.gov/ccg/research/genome-sequencing/tcga">TCGA</a>
    </tr>
    <tr>
        <td>GBM/LGG</td>
        <td>
        Merged dataset from two types of brain cancer: Glioblastoma Multiforme and Lower Grade Glioma.
        </td>
        <td><a href="https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr">Chen et al.</a>
    </tr>
    <tr>
        <td>KIRC</td>
        <td>
        Kidney Renal Clear Cell Carcinoma.
        </td>
        <td><a href="https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr">Chen et al.</a>
    </tr>
</table>

# Reproducibility

All results in the associated paper can be reproduced using the codes in `reprod/`. Please refer to the instructions in `reprod/README.md`. 

# Credits

Special thanks to:
- All authors of Kolmogorov-Arnold Networks and the incredible [pykan](https://github.com/KindXiaoming/pykan) package. 
- Håvard Kvamme for [pycox](https://github.com/havakv/pycox) and [torchtuples](https://github.com/havakv/torchtuples).

