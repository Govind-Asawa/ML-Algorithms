import numpy as np
import random
import time
import copy

from sklearn.tree import DecisionTreeRegressor

class CustomBaggingRegressor(object):

    def __init__(self, base_estimator = None ,n_estimators=10, max_samples = 0.6):
        """
            :params:
                base_estimator - base estimator to be used to fit onto sub samples
                n_estimators - no of base estimators to fit
                max_samples - float ,% of samples to be included in each sub-sample used to train respec
                            estimators
        """
        self.base_estimator = base_estimator if base_estimator else DecisionTreeRegressor()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators_samples = []
        self.estimators = []
    
    def bootstrap(self, X, y):
        """
            generator function which yeilds one sub-sample (x, y) at a time
            i.e., it implements bootstraping
        """
        np.random.seed(1)
        random.seed(1)
        
        for i in range(self.n_estimators):
            self.estimators_samples.append(random.sample(range(X.shape[0]), int(X.shape[0] * self.max_samples)))
            yield ( X[self.estimators_samples[i], :], y[self.estimators_samples[i]] )
    
    def predict(self, X):
        return np.mean(np.array([estimator.predict(X) for estimator in self.estimators]), axis=0)
    
    def cost(self, y, pred_y):
        """
            :return:
                MSE (mean squared error)
        """
        return np.mean((y-pred_y)**2)

    def fit(self, X, y):

        for i, (subsample_X, subsample_y) in enumerate(self.bootstrap(X, y)):
            self.estimators.append(copy.deepcopy(self.base_estimator))
            self.estimators[i].fit(subsample_X, subsample_y)
