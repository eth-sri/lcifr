from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class LawschoolDataset(AbstractDataset):
    def __init__(self, split, args, normalize=True):
        super().__init__('lawschool', split)

        data_file = path.join(self.data_dir, 'lawschs1_1.dta')

        if not path.exists(data_file):
            request.urlretrieve(
                'http://www.seaphe.org/databases/FOIA/lawschs1_1.dta', data_file
            )

        dataset = pd.read_stata(data_file)
        dataset.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
        dataset.dropna(axis=0, inplace=True, subset=['admit'])
        dataset.replace(to_replace='', value=np.nan, inplace=True)
        dataset.dropna(axis=0, inplace=True)
        dataset = dataset[dataset['race'] != 'Asian']

        features, labels = dataset.drop('admit', axis=1), dataset['admit']

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

        continuous_vars.remove('gender')
        self.categorical_columns.append('gender')

        self.protected_unique = 2
        protected = (features['race'] != 'White')

        features = pd.get_dummies(features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

        if args.quantiles:
            features['quantiles'] = features.groupby('race=White')['gpa'].rank('dense', ascending=False)
            features.drop('gpa', axis=1, inplace=True)
            self.continuous_columns.append(features.columns.get_loc('quantiles'))

        self.one_hot_columns = {}
        for column_name in self.categorical_columns:
            ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        print('categorical features: ', self.one_hot_columns.keys())

        self.column_ids = {col: idx for idx, col in enumerate(features.columns)}

        features = torch.tensor(features.values.astype(np.float32), device=self.device)
        labels = torch.tensor(labels.values.astype(np.int64), device=self.device)
        protected = torch.tensor(protected.values.astype(np.bool), device=self.device)

        X_train, self.X_test, y_train, self.y_test, protected_train, self.protected_test = train_test_split(
            features, labels, protected, test_size=0.2, random_state=0
        )
        self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
            X_train, y_train, protected_train, test_size=0.2, random_state=0
        )

        if normalize:
            self._normalize(self.continuous_columns)

        self._assign_split()
