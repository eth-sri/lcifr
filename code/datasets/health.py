"""
Preprocessing based on: https://github.com/truongkhanhduy95/Heritage-Health-Prize
"""
import zipfile
from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class HealthDataset(AbstractDataset):
    column_names = ['MemberID', 'ProviderID', 'Sex', 'AgeAtFirstClaim']
    claims_cat_names = ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc']

    def __init__(self, split, args, normalize=True):
        super().__init__('health', split)

        health_file = path.join(self.data_dir, 'health_full.csv')

        if not (args.load and path.exists(health_file)):
            data_file = path.join(self.data_dir, 'HHP_release3.zip')

            if not path.exists(data_file):
                request.urlretrieve('https://foreverdata.org/1015/content/HHP_release3.zip', data_file)

            zf = zipfile.ZipFile(data_file)

            df_claims = self.preprocess_claims(pd.read_csv(zf.open('Claims.csv'), sep=','))
            df_drugs = self.preprocess_drugs(pd.read_csv(zf.open('DrugCount.csv'), sep=','))
            df_labs = self.preprocess_labs(pd.read_csv(zf.open('LabCount.csv'), sep=','))
            df_members = self.preprocess_members(pd.read_csv(zf.open('Members.csv'), sep=','))

            df_labs_drugs = pd.merge(df_labs, df_drugs, on=['MemberID', 'Year'], how='outer')
            df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=['MemberID', 'Year'], how='outer')
            df_health = pd.merge(df_labs_drugs_claims, df_members, on=['MemberID'], how='outer')

            df_health.drop(['Year', 'MemberID'], axis=1, inplace=True)
            df_health.fillna(0, inplace=True)

            df_health.to_csv(health_file, index=False)

        df_health = pd.read_csv(health_file, sep=',')

        if args.label:
            labels = df_health[args.label]
        else:
            labels = 1 - df_health['max_CharlsonIndex']

        if args.transfer:
            drop_cols = [col for col in df_health.columns if col.startswith('PrimaryConditionGroup=')]
            df_health.drop(drop_cols, axis=1, inplace=True)

        features = df_health.drop('max_CharlsonIndex', axis=1)

        continuous_vars = [col for col in features.columns if '=' not in col]
        self.continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

        self.protected_unique = 2
        protected = np.logical_or(
            features['AgeAtFirstClaim=60-69'], np.logical_or(
                features['AgeAtFirstClaim=70-79'], features['AgeAtFirstClaim=80+']
            )
        )

        self.one_hot_columns = {}
        for column_name in HealthDataset.column_names:
            ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        print('categorical features: ', self.one_hot_columns.keys())

        self.column_ids = {col: idx for idx, col in enumerate(features.columns)}

        features = torch.tensor(features.values.astype(np.float32), device=self.device)
        labels = torch.tensor(labels.values.astype(np.int64), device=self.device).bool().long()
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

    @staticmethod
    def preprocess_claims(df_claims):
        df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
        df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

        df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
        df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
        df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
        df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
        df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
        df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
        df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
        df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
        df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
        df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
        df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
        df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

        df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
        df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
        df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
        df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

        df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
        df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
        df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
        df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
        df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
        df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
        df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
        df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
        df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
        df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
        df_claims['LengthOfStay'].fillna(0, inplace=True)
        df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

        for cat_name in HealthDataset.claims_cat_names:
            df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
        df_claims = pd.get_dummies(df_claims, columns=HealthDataset.claims_cat_names, prefix_sep='=')

        oh = [col for col in df_claims if '=' in col]

        agg = {
            'ProviderID': ['count', 'nunique'],
            'Vendor': 'nunique',
            'PCP': 'nunique',
            'CharlsonIndex': 'max',
            # 'PlaceSvc': 'nunique',
            # 'Specialty': 'nunique',
            # 'PrimaryConditionGroup': 'nunique',
            # 'ProcedureGroup': 'nunique',
            'PayDelay': ['sum', 'max', 'min']
        }
        for col in oh:
            agg[col] = 'sum'

        df_group = df_claims.groupby(['Year', 'MemberID'])
        df_claims = df_group.agg(agg).reset_index()
        df_claims.columns = [
                                'Year', 'MemberID', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs',
                                'max_CharlsonIndex', 'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                            ] + oh

        return df_claims

    @staticmethod
    def preprocess_drugs(df_drugs):
        df_drugs.drop(columns=['DSFS'], inplace=True)
        # df_drugs['DSFS'] = df_drugs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
        df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
        df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
        print('df_drugs.shape = ', df_drugs.shape)
        return df_drugs

    @staticmethod
    def preprocess_labs(df_labs):
        df_labs.drop(columns=['DSFS'], inplace=True)
        # df_labs['DSFS'] = df_labs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
        df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
        df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
        print('df_labs.shape = ', df_labs.shape)
        return df_labs

    @staticmethod
    def preprocess_members(df_members):
        df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
        df_members['Sex'].fillna('?', inplace=True)
        df_members = pd.get_dummies(
            df_members, columns=['AgeAtFirstClaim', 'Sex'], prefix_sep='='
        )
        print('df_members.shape = ', df_members.shape)
        return df_members
