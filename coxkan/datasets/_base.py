from pathlib import Path
import os
import coxkan

DATA_DIR_OVERRIDE = os.environ.get('COXKAN_DATA_DIR', None)
if DATA_DIR_OVERRIDE:
    DATA_DIR = Path(DATA_DIR_OVERRIDE)
else:
    DATA_DIR = Path(coxkan.__file__).parent / 'datasets' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

class _BaseDataset:
    """Abstract class for loading data sets.
    """
    name = NotImplemented

    def __init__(self):
        self.path = DATA_DIR / f"{self.name}.feather"
        self.path_train = DATA_DIR / f"{self.name}_train.feather"
        self.path_test = DATA_DIR / f"{self.name}_test.feather"

    def load_df(self):
        raise NotImplementedError
    
    def _download(self):
        raise NotImplementedError
    
    def delete_local_copy(self):
        if self.path.exists(): self.path.unlink()
        if self.path_train.exists(): self.path_train.unlink()
        if self.path_test.exists(): self.path_test.unlink()
        else:
            raise RuntimeError("Local copy does not exist.")

    def _label_cols_at_end(self, df):
        if hasattr(self, 'duration_col') and hasattr(self, 'event_col'):
            label_col = [self.duration_col, self.event_col]
            df = df[list(df.columns.drop(label_col)) + label_col]
        return df
    
    def metadata(self):
        return self.name, self.duration_col, self.event_col, self.covariates

    def __str__(self):
        return self.__doc__
