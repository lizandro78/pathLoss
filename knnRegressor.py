# Importando as libs

import numpy as np
from sklearn import base as skBase
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#Biblioteca do K-NN


# Regressor Multiplo

class MultiRegressor(skBase.BaseEstimator):
    #A classe MultiRegressor herda de skBase.BaseEstimator
    #Base class for all estimators in scikit-learn

    def __init__(self, estimator, cv_parms):
        self.estimator = estimator
        self.scaler = StandardScaler()
        self.cv_parms = cv_parms
        self.best_parameters_ = []

    def fit(self, X, y):
        X_t = self.scaler.fit_transform(X)
        n, m = y.shape
        #n linhas e m colunas
        self.estimators_ = []
        for i in range(m):
            grid = GridSearchCV(self.estimator, self.cv_parms, cv=10, scoring='neg_mean_squared_error')
            grid.fit(X_t, y.iloc[:, i])
            self.estimators_.append(grid)
            self.best_parameters_.append(grid.best_params_)
        return self

    def predict(self, X):
        X_t = self.scaler.transform(X)
        res = [est.predict(X_t)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)
