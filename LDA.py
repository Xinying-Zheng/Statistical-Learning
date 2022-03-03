# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:33:21 2021

@author: xinying-zheng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def getData(filename, labels):
    print('loading data ...')
    df = pd.read_csv(filename)
    X = []
    Y = []
    splits = []
    for i, label in enumerate(labels):
        rows = np.where(df['label'] == label)
        data = np.array([df.iloc[row][1:].tolist() for row in rows[0]])
        X = np.vstack((X, data)) if i > 0 else data
        Y = np.hstack((Y, np.array([label] * data.shape[0]))) if i > 0 \
            else np.array([label] * data.shape[0])
        splits.append(X.shape[0])
    
    X = X / 255
    print('finish loading ...')
    return X, Y, splits

def confusion_M(Y, predictions, N):
    n = Y.shape[0]
    res = np.zeros((N, N))
    for i in range(n):
        res[predictions[i]][int(Y[i])] += 1
    
    return res

def plot_M(M, title):
    f, ax = plt.subplots()
    sn.heatmap(M, annot=True, ax=ax)
    ax.set_title(title)
    plt.show()
    
class LDA:
    def __init__(self):
        self.parameters = {}
    
    def fit(self, X, Y, splits):
        num_samples, feature_len = X.shape
        sigma = np.zeros((feature_len, feature_len))
        k_cls = len(splits)
        
        begin = 0
        for i, end in enumerate(splits):
            X_i = X[begin:end]
            # compute the mean array
            mean_i = np.mean(X_i, axis=0)
            # compute the propotion
            pi_i = (end - begin) / num_samples
            # compute the autovariance matrix
            sigma += (X_i - mean_i).T.dot(X_i - mean_i)

            begin = end
            self.parameters[i] = {'mean' : mean_i, 'log pi' : np.log(pi_i)}
        
        inverse_sigma = np.linalg.pinv(sigma / (num_samples - k_cls))
        
        for i in self.parameters:
            self.parameters[i]['second term'] = 0.5 * self.parameters[i]['mean'].dot(inverse_sigma).dot(self.parameters[i]['mean'])
        
        self.parameters['inverse sigma'] = inverse_sigma
        print(self.parameters.keys())
        
    def predict(self, X):
        res = []
        def criteria(X, key):
            # generate the criteria
            return X.dot(self.parameters['inverse sigma']).dot(self.parameters[key]['mean']) - \
                self.parameters[key]['second term'] + self.parameters[key]['log pi']
                
        for i, key in enumerate(self.parameters):
            if key == 'inverse sigma':
                break
            # compute the value of each criteria
            res = criteria(X, key) if i == 0 else np.vstack((res, criteria(X, key)))
            # print(key, res.shape)
            
        return np.argmax(res, axis=0)
    
if __name__ == '__main__':
    X, Y, split_train = getData('train.csv', [0, 1, 2])
    X_test, Y_test, split_test = getData('test.csv', [0, 1, 2])
    predictor = LDA()
    predictor.fit(X, Y, split_train)
    predictions = predictor.predict(X)

    M = confusion_M(Y, predictions, 3)
    plot_M(M, 'On training set')
    
    predictions = predictor.predict(X_test)
    M = confusion_M(Y_test, predictions, 3)
    plot_M(M, 'On testing set')
