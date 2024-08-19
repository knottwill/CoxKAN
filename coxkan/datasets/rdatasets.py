import pandas as pd
from ._base import _BaseDataset
from sklearn.model_selection import train_test_split

def download_from_rdatasets(package, name):
    datasets = (pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/datasets.csv")
                .loc[lambda x: x['Package'] == package].set_index('Item'))
    if not name in datasets.index:
        raise ValueError(f"Dataset {name} not found.")
    info = datasets.loc[name]
    url = info.CSV
    return pd.read_csv(url), info

class FLCHAIN(_BaseDataset):
    """
    Assay of serum free light chain (FLCHAIN).
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    A study of the relationship between serum free light chain (FLC) and mortality.
    The original sample contains samples on approximately 2/3 of the residents of Olmsted
    County aged 50 or greater.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html

    Variables:
        age:
            age in years.
        sex:
            F=female, M=male.
        sample.yr:
            the calendar year in which a blood sample was obtained.
        kappa:
            serum free light chain, kappa portion.
        lambda:
            serum free light chain, lambda portion.
        flc.grp:
            the FLC group for the subject, as used in the original analysis.
        creatinine:
            serum creatinine.
        mgus:
            1 if the subject had been diagnosed with monoclonal gammapothy (MGUS).
        futime: (duration)
            days from enrollment until death. Note that there are 3 subjects whose sample
            was obtained on their death date.
        death: (event)
            0=alive at last contact date, 1=dead.
        chapter:
            for those who died, a grouping of their primary cause of death by chapter headings
            of the International Code of Diseases ICD-9.

    """
    name = 'flchain'
    duration_col = 'futime'
    event_col = 'death'
    covariates = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']
    categorical_covariates = ['sex', 'sample.yr', 'flc.grp', 'mgus']

    def load(self, split=False):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.
        """

        if not self.path_train.exists() and not self.path_test.exists():
            print(f"Downloading dataset '{self.name}' ...")
            df, info = download_from_rdatasets('survival', self.name)
            self.info = info
            df = self._label_cols_at_end(df)

            df = (df
                    .drop(['chapter', 'rownames'], axis=1)
                    .loc[lambda x: x['creatinine'].isna() == False]
                    .reset_index(drop=True))

            for col in self.categorical_covariates:
                df[col] = df[col].astype('category')
            for col in df.columns.drop(self.categorical_covariates):
                df[col] = df[col].astype('float32')

            df_train, df_test = train_test_split(df, test_size=0.2, random_state=2024)
            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)
            df_train.to_feather(self.path_train)
            df_test.to_feather(self.path_test)
            print('Done.')
        else:
            df_train = pd.read_feather(self.path_train)
            df_test = pd.read_feather(self.path_test)

        if split:
            return df_train, df_test

        return pd.concat([df_train, df_test], ignore_index=True)


class NWTCO(_BaseDataset):
    """
    Data from the 3rd and 4th clinical trails National Wilm's Tumor Study Group
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    Measurement error example. Tumor histology predicts survival, but prediction is stronger
    with central lab histology than with the local institution determination.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html

    Variables:
        instit:
            histology reading from local institution:
                - 1: favorable
                - 2: unfavorable
        histol:
            histology reading from central lab:
                - 1: favorable
                - 2: unfavorable
        stage:
            disease stage: 
                - 1: localized to the kidney and completely resected
                - 2: spread beyond thekidney but completely resected
                - 3: residual tumour in the abdomen or tumour in the lymphnodes
                - 4: metastatic to the lung or liver.
        study:
            clinical trial number (3 or 4)
        age:
            age in months
        in.subcohort:
            included in the subcohort for the example in the paper
        rel: (event)
            indicator for relapse
        edrel: (duration)
            time to relapse

    References
        NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary
        outcome applied to Wilms tumor prognosis. Applied Statistics 48, 457–68.
    """
    name = 'nwtco'
    duration_col = 'edrel'
    event_col = 'rel'
    covariates = ['instit', 'histol', 'stage', 'study', 'age', 'in.subcohort']
    categorical_covariates = ['instit', 'histol', 'stage', 'study', 'in.subcohort']

    def load(self, split=False):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.
        """

        if not self.path_train.exists() and not self.path_test.exists():
            print(f"Downloading dataset '{self.name}' ...")
            df, info = download_from_rdatasets('survival', self.name)
            self.info = info
            df = self._label_cols_at_end(df)

            df = (df.drop(['rownames', 'seqno'], axis=1))

            for col in self.categorical_covariates:
                df[col] = df[col].astype('category')
            for col in df.columns.drop(self.categorical_covariates):
                df[col] = df[col].astype('float32')

            df_train, df_test = train_test_split(df, test_size=0.2, random_state=2024)
            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)
            df_train.to_feather(self.path_train)
            df_test.to_feather(self.path_test)
            print('Done.')

        else:
            df_train = pd.read_feather(self.path_train)
            df_test = pd.read_feather(self.path_test)

        if split:
            return df_train, df_test

        return pd.concat([df_train, df_test], ignore_index=True)
