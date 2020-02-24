from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class GermanDataset(AbstractDataset):
    column_names = [
        'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
        'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
        'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
        'foreign_worker', 'credit'
    ]
    personal_status_map = {'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'}

    def __init__(self, split, args, normalize=True):
        super().__init__('german', split)

        data_file = path.join(self.data_dir, 'german.data')

        if not path.exists(data_file):
            request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', data_file
            )

        dataset = pd.read_csv(data_file, sep=' ', header=None, names=GermanDataset.column_names)
        dataset['sex'] = dataset['personal_status'].replace(GermanDataset.personal_status_map)
        dataset.drop('personal_status', axis=1, inplace=True)
        features, labels = dataset.drop('credit', axis=1), dataset['credit']

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

        protected_att = args.protected_att if args.protected_att is not None else 'sex'
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
        labels = 2 - torch.tensor(labels.values.astype(np.int64), device=self.device)
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
