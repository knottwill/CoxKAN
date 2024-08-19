from collections import defaultdict
import requests
import h5py
import pandas as pd
from ._base import _BaseDataset, DATA_DIR

DEEPSURV_URL = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"

DEEPSURV_DATASETS = {
    'support': "support/support_train_test.h5",
    'metabric': "metabric/metabric_IHC4_clinical_train_test.h5",
    'gbsg': "gbsg/gbsg_cancer_train_test.h5",
    'whas': "whas/whas_train_test.h5"
}

def download_from_deepsurv(name, covariates=None):
    url = DEEPSURV_URL + DEEPSURV_DATASETS[name]
    path = DATA_DIR / f"{name}.h5"
    with requests.Session() as s:
        r = s.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

    data = defaultdict(dict)
    with h5py.File(path) as f:
        for ds in f: 
            for array in f[ds]:
                data[ds][array] = f[ds][array][:]

    path.unlink()
    assert 'valid' not in data, f"Dataset {name} has a validation set."

    df_train = _make_df(data['train'], covariates)
    df_test = _make_df(data['test'], covariates)
    return df_train, df_test

def _make_df(data, covariates=None):
    x = data['x']
    t = data['t']
    e = data['e']

    if covariates is None:
        covariates = ['x'+str(i) for i in range(x.shape[1])]

    df = (pd.DataFrame(x, columns=covariates)
          .assign(duration=t)
          .assign(event=e))
    return df

class _DeepSurvDataset(_BaseDataset):
    duration_col = 'duration'
    event_col = 'event'
    def _download(self):
        df_train, df_test = download_from_deepsurv(self.name, self.covariates)
        df_train.to_feather(self.path_train)
        df_test.to_feather(self.path_test)

    def load(self, split=False):
        df_train, df_test = None, None
        if not self.path_train.exists() and not self.path_test.exists():
            print(f"Downloading dataset '{self.name}' from DeepSurv ...")
            self._download()
            print(f"Done")

        df_train = pd.read_feather(self.path_train)
        df_test = pd.read_feather(self.path_test)
        df_train = self._label_cols_at_end(df_train)
        df_test = self._label_cols_at_end(df_test)

        for cat in self.categorical_covariates:
            df_train[cat] = df_train[cat].astype('category')
            df_test[cat] = df_test[cat].astype('category')

        if split:
            print('Using default train-test split (used in DeepSurv paper).')
            return df_train, df_test
        else:
            return pd.concat([df_train, df_test], ignore_index=True)


class SUPPORT(_DeepSurvDataset):
    """
    Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).

    A study of survival for seriously ill hospitalized adults.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Covariate names restored from https://hbiostat.org/data/repo/support2csv.zip

    Variables:
        age:
            age in years.
        sex:
            patient sex. (1: female, 0: male)
        race:
            patient race (unfortunately, the original data did not provide which the values represent which races).
        comorbidity:
            number of comorbidities.
        diabetes:
            presence of diabetes.
        dementia:
            presence of dementia.
        cancer:
            presence of cancer. (2: yes, 1: no, 0: metastatic)
        meanbp:
            mean arterial blood pressure.
        hr:
            heart rate.
        rr:
            respiration rate.
        temp:
            temperature.
        sodium:
            serum’s sodium.
        wbc:
            white blood cell count.
        creatinine:
            serum’s creatinine.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            death indicator {1: death, 0: censoring}.
    """
    name = 'support'
    covariates = ['age', 'sex', 'race', 'comorbidity', 'diabetes', 'dementia', 'cancer', 'meanbp', 'hr', 'rr', 'temp', 'sodium', 'wbc', 'creatinine']
    categorical_covariates = ['sex', 'race', 'diabetes', 'dementia', 'cancer']


class METABRIC(_DeepSurvDataset):
    """
    The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).

    Gene and protein expression profiles to determine new breast cancer subgroups in
    order to help physicians provide better treatment recommendations.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    According to the DeepSurv paper, the data was preprocessed according to the 
    Immunohistochemical 4 plus Clinical (IHC4+C) test, such that the first four covariates
    are gene expression indicators and the last five are clinical features.

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Covariate names restored from https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
    (For the gene expressions, we calculate z-score to compare them)

    Variables:
        EGFR:
            Epidermal growth factor receptor.
        PGR:
            Progesterone receptor.
        ERBB2:
            Human epidermal growth factor receptor 2.
        MKI67:
            Ki-67 protein expression.
        hormone:
            Hormone treatment indicator.
        radio:
            Radiotherapy indicator.
        chemo:
            Chemotherapy indicator.
        ER:
            Estrogen receptor positive indicator.
        age:
            Age at diagnosis.
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """
    name = 'metabric'
    covariates = ['EGFR', 'PGR', 'ERBB2', 'MKI67', 'hormone', 'radio', 'chemo', 'ER', 'age']
    categorical_covariates = ['hormone', 'radio', 'chemo', 'ER']


class GBSG(_DeepSurvDataset):
    """ 
    Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.

    Covariate names restored from https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman

    Variables:
        hormon
            hormonal therapy, 0= no, 1= yes
        size
            tumor size (0: <20 mm, 1: [20 mm to 50 mm], 2: > 50 mm))
        meno
            menopausal status (0= premenopausal, 1= postmenopausal)
        age
            age, years
        nodes
            number of positive lymph nodes
        pgr
            progesterone receptors (fmol/l)
        er
            estrogen receptors (fmol/l)
        duration: (duration)
            the right-censored event-times.
        event: (event)
            event indicator {1: event, 0: censoring}.
    """
    name = 'gbsg'
    covariates = ['hormon', 'size', 'meno', 'age', 'nodes', 'pgr', 'er']
    categorical_covariates = ['hormon', 'size', 'meno']