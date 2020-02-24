from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class CompasDataset(AbstractDataset):

    def __init__(self, split, args, normalize=True):
        super().__init__('compas', split)

        datafile = path.join(self.data_dir, 'compas-scores-two-years.csv')

        if not path.exists(datafile):
            request.urlretrieve(
                'https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv', datafile
            )

        df = pd.read_csv(datafile)
        df = df[df['days_b_screening_arrest'] >= -30]
        df = df[df['days_b_screening_arrest'] <= 30]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != '0']
        df = df[df['score_text'] != 'N/A']

        df['in_custody'] = pd.to_datetime(df['in_custody'])
        df['out_custody'] = pd.to_datetime(df['out_custody'])
        df['diff_custody'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['diff_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()

        df.drop(
            [
                'id', 'name', 'first', 'last', 'v_screening_date', 'compas_screening_date', 'dob', 'c_case_number',
                'screening_date', 'in_custody', 'out_custody', 'c_jail_in', 'c_jail_out'
            ], axis=1, inplace=True
        )
        df = df[df['race'].isin(['African-American', 'Caucasian'])]

        features = df.drop(['is_recid', 'is_violent_recid', 'violent_recid', 'two_year_recid'], axis=1)
        labels = 1 - df['two_year_recid']

        features = features[[
            'age', 'sex', 'race', 'diff_custody', 'diff_jail', 'priors_count', 'juv_fel_count', 'c_charge_degree',
            'c_charge_desc', 'v_score_text'
        ]]

        continuous_vars = []
        self.categorical_columns = []
        for col in features.columns:
            if features[col].isnull().sum() > 0:
                features.drop(col, axis=1, inplace=True)
            else:
                if features[col].dtype == np.object:
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]

        protected_att = args.protected_att if args.protected_att is not None else 'race'
        self.protected_unique = features[protected_att].nunique()
        protected = np.logical_not(pd.Categorical(features[protected_att]).codes)

        features = pd.get_dummies(features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

        self.one_hot_columns = {}
        for column_name in self.categorical_columns:
            ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        print('categorical features: ', self.one_hot_columns.keys())

        self.column_ids = {col: idx for idx, col in enumerate(features.columns)}

        features = torch.tensor(features.values.astype(np.float32), device=self.device)
        labels = torch.tensor(labels.values.astype(np.int64), device=self.device).bool().long()
        protected = torch.tensor(protected, device=self.device).bool()

        X_train, self.X_test, y_train, self.y_test, protected_train, self.protected_test = train_test_split(
            features, labels, protected, test_size=0.2, random_state=0
        )
        self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
            X_train, y_train, protected_train, test_size=0.2, random_state=0
        )

        if normalize:
            self._normalize(self.continuous_columns)

        self._assign_split()
