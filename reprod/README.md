# Reproducibility 

To re-produce all results quoted in the paper, follow the steps below:

(It is assumed you are running all commands from the project root directory)

### 1. Obtain data and checkpoints

Set the environment variable `COXKAN_DATA_DIR` to the `data/` directory in the root of the project. Locate your shell configuration file (If you are using bash that will be `~/.bashrc` or `~/.bash_profile`), and add the absolute path to the data directory:

```bash
export COXKAN_DATA_DIR="path/to/data/"
```

The TCGA genomics data was too large to include in the repo; please request it by emailing knottenbeltwill@gmail.com. I can also send you all my model checkpoints / results. 

### 2. Re-create environment

```bash
pip install -r reprod/requirements.txt
```

### 3. Optional: Perform hyper-parameter searches

You can skip this step by just using the hyper-parameters from my searches in `configs/`

Hyper-parameter searches for CoxKAN:
```bash
python reprod/sweep.py --exp_name sim_gaussian
python reprod/sweep.py --exp_name sim_depth_1
python reprod/sweep.py --exp_name sim_deep --config configs/sweep/coxkan_deep.yml
python reprod/sweep.py --exp_name sim_difficult
python reprod/sweep.py --exp_name support
python reprod/sweep.py --exp_name gbsg
python reprod/sweep.py --exp_name metabric
python reprod/sweep.py --exp_name nwtco
python reprod/sweep.py --exp_name flchain
python reprod/sweep.py --exp_name TCGA-STAD --config configs/sweep/coxkan_deep.yml
python reprod/sweep.py --exp_name TCGA-BRCA --config configs/sweep/coxkan_deep.yml
python reprod/sweep.py --exp_name TCGA-GBMLGG --config configs/sweep/coxkan_deep.yml
python reprod/sweep.py --exp_name TCGA-KIRC --config configs/sweep/coxkan_deep.yml
```

Hyper-parameter searches for DeepSurv (MLP-based model):

```bash
python reprod/sweep.py --exp_name nwtco --model mlp
python reprod/sweep.py --exp_name flchain --model mlp
python reprod/sweep.py --exp_name TCGA-STAD --model mlp
python reprod/sweep.py --exp_name TCGA-BRCA --model mlp
python reprod/sweep.py --exp_name TCGA-GBMLGG --model mlp
python reprod/sweep.py --exp_name TCGA-KIRC --model mlp
```

### 4. Synthetic datasets (Simulation studies)

Move `simulation.ipynb` to the root directory and run all cells.

### 5. First 3 clinical datasets

```bash
python reprod/clinical --exp_name metabric
python reprod/clinical --exp_name gbsg
python reprod/clinical --exp_name flchain
```

### 6. Clinical datasets with extra interpretability: SUPPORT and NWTCO

The analysis for SUPPORT and NWTCO is done in notebooks since it involves interaction terms which we plot / interpret.

Move the `support.ipynb` and `nwtco.ipynb` notebooks to the root directory and run them.

### 7. Genomic datasets

Move `genomics.ipynb` to the root directory and follow the instructions inside.