from os import defpath
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from configs import features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import os
import warnings
warnings.filterwarnings(action='ignore')

class MyDataController():
    def __init__(self, base_df, nsplit=5):
        self.base_df = base_df
        self.nsplit = 5
        self.df = self._get_featured_dataframe()

    def _get_featured_dataframe(self):
        indexs = self.base_df[features].dropna(axis=0).index
        df = self.base_df.loc[indexs].reset_index(drop=True)
        # convert string to int object
        subject_uniques = {key: i for i, key in enumerate(
            set(df['Subject'].to_list()))}
        for i in range(len(df)):
            df.loc[i, 'Subject'] = subject_uniques[df.loc[i, 'Subject']]

        return df

    def __len__(self):
        return len(self.df)

    def get_dataset(self, selected_features: list[str], scaling=True, sampling=True):
        X = []
        Y = []
        ID = []
        for i in range(len(self.df)):
            X.append(self.df.loc[i, selected_features].to_list())
            Y.append(self.df.loc[i, 'Sensation'])
            ID.append(self.df.loc[i,'ID'])
        # scaler
        if scaling == True:
            X = self.scaling(X)
        
        # generate train set and test set with K-Fold
        X_trainSet, X_testSet, y_trainSet, y_testSet, id_trainSet, id_testSet = [], [], [], [], [], []
        skf = StratifiedKFold(n_splits=self.nsplit, shuffle=True)
        for train_index, test_index in skf.split(X, Y):
            X_train = [X[i] for i in train_index]
            X_test = [X[i] for i in test_index]
            y_train = [Y[i] for i in train_index]
            y_test = [Y[i] for i in test_index]
            id_train = [ID[i] for i in train_index]
            id_test = [ID[i] for i in test_index]
            
            y_classes, _ = np.unique(y_train, return_counts=True)

            # sampling
            if sampling == True:
                X_train, y_train = self.sampling(X_train, y_train)

            X_trainSet.append(X_train)
            X_testSet.append(X_test)
            y_trainSet.append(y_train)
            y_testSet.append(y_test)
            id_trainSet.append(id_train)
            id_testSet.append(id_test)
        print('-'*30)
        print('Total number of data: ', np.array(X).shape[0])
        print('Train - {} ({}/{}), Test - {} ({}/{})'.format(
            np.array(X_trainSet[0]).shape[0], self.nsplit-1, self.nsplit,
            np.array(X_testSet[0]).shape[0], 1, self.nsplit, ))
        print('{} selected features: {}'.format(len(selected_features), selected_features))
        print('Standard Scaling: {}\tSMOTE Sampling: {}'.format(scaling, sampling))
        
        return X, Y, X_trainSet, X_testSet, y_trainSet, y_testSet, id_trainSet, id_testSet
        
    def scaling(self, X):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        return X

    def sampling(self, X_train, y_train):
        y_classes, _ = np.unique(y_train, return_counts=True)

        SMOTE_dict = {}
        for y_class in y_classes:
            SMOTE_dict[y_class] = int(len(y_train)/len(y_classes))

        over = SMOTE(sampling_strategy=SMOTE_dict)
        under = RandomUnderSampler()
        pipeline = Pipeline(steps=[('u', under), ('o', over)])
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

        return X_train, y_train
    
    def plot_feature(self, selected_features):

        for i , name in enumerate(selected_features):
            X = self.df[name].to_numpy()
            # calculate bin width using Freedman-Diaconis rule
            iqr = np.percentile(X, 75) - np.percentile(X, 25)
            bin_width = 2 * iqr / np.shape(X)[0] ** (1/3)
            # plot histogram with selected bin width
            if bin_width == 0:
                plt.hist(X, bins=10, edgecolor='black')
            else:
                plt.hist(X, bins=int((X.max() - X.min()) / bin_width), edgecolor='black')
                
            plt.title(name)
            plt.show()


if __name__ == "__main__":
    df = pd.read_csv('D:/ThermalData/Charlotte_ThermalFace/S_all_temp.csv', index_col=0)
    nsplit = 5
    dataController = MyDataController(df, nsplit)
    selected_features = features
    dataController.plot_feature(selected_features)
    
    X, Y, X_trainSet, X_testSet, y_trainSet, y_testSet, id_trainSet, id_testSet = dataController.get_dataset(selected_features)
