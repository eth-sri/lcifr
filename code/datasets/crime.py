from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class CrimeDataset(AbstractDataset):
    column_names = [
        'communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack',
        'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
        'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
        'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
        'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
        'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
        'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar',
        'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
        'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
        'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
        'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
        'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
        'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ',
        'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters',
        'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
        'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
        'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
        'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
        'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
        'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders',
        'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
        'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
        'ViolentCrimesPerPop', 'nonViolPerPop'
    ]

    def __init__(self, split, args, normalize=True):
        super().__init__('crime', split)

        data_file = path.join(self.data_dir, 'communities.data')

        if not path.exists(data_file):
            request.urlretrieve(
                'http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt',
                data_file
            )

        dataset = pd.read_csv(data_file, sep=',', header=None, names=CrimeDataset.column_names)
        # remove features that are not predictive
        dataset.drop(['communityname', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)
        # remove all other potential goal variables
        dataset.drop(
            [
                'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
                'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
                'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop'
            ], axis=1, inplace=True
        )
        dataset.replace(to_replace='?', value=np.nan, inplace=True)
        # drop rows with missing labels
        dataset.dropna(axis=0, subset=['ViolentCrimesPerPop'], inplace=True)
        # drop columns with missing values
        dataset.dropna(axis=1, inplace=True)
        features, labels = dataset.drop('ViolentCrimesPerPop', axis=1), dataset['ViolentCrimesPerPop']

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

        self.protected_unique = 2
        protected = np.less(
            features['racePctWhite'] / 5, features['racepctblack'] + features['racePctAsian'] + features['racePctHisp']
        )

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
        labels = torch.tensor(labels.values.astype(np.float32), device=self.device)
        protected = torch.tensor(protected.values.astype(np.bool), device=self.device)

        # binarize labels
        labels = (labels < labels.median())

        X_train, self.X_test, y_train, self.y_test, protected_train, self.protected_test = train_test_split(
            features, labels, protected, test_size=0.2, random_state=0
        )
        self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
            X_train, y_train, protected_train, test_size=0.2, random_state=0
        )

        if normalize:
            self._normalize(self.continuous_columns)

        self._assign_split()
