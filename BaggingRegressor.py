import multiprocessing as mp
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
    
    def fitEstimator(self, estimator, subsample_X, subsample_y):
        return estimator.fit(subsample_X, subsample_y)

    def __bootstrap(self, X, y):
        """
            generator function which yeilds one sub-sample (x, y) at a time
            i.e., it implements bootstraping
        """
        np.random.seed(1)
        random.seed(1)
        
        for i in range(self.n_estimators):
            self.estimators_samples.append(random.sample(range(X.shape[0]), int(X.shape[0] * self.max_samples)))
            yield ( X[self.estimators_samples[i], :], y[self.estimators_samples[i]] )
    
    def predictEstimator(self, estimator, X):
        return estimator.predict(X)

    def predict(self, X):
        """
        predicts the target variable for given X 
        by applying parallel processing
            :params:
                X - 2D array of size (m, n) where 
                    m is the no of samples and 
                    n is the no of features
            :return:

        """
        pool = mp.Pool(mp.cpu_count())

        predictions = np.array(
            pool.starmap(
                self.predictEstimator, [ (estimator, X) for estimator in self.estimators ]
            )
        )

        return np.mean(predictions, axis=0)
    
    def cost(self, y, pred_y):
        """
            :return:
                MSE (mean squared error)
        """
        return np.mean((y-pred_y)**2)

    def fit(self, X, y):
        """
            function to fit n_estimators for the given X, y
            by applying parallel processing (uses all the cores available)

            :params:
                X - 2D array of features 
                Y - 1D array of target variable
            
            :return:
                None
        """
        pool = mp.Pool(mp.cpu_count())
        
        self.estimators = pool.starmap(self.fitEstimator,
                                        [ (copy.deepcopy(self.base_estimator), subsample_X, subsample_y) 
                                        for (subsample_X, subsample_y) in self.__bootstrap(X, y)])
        
        pool.close()
        # for i, (subsample_X, subsample_y) in enumerate(self.__bootstrap(X, y)):
        #     self.estimators.append(copy.deepcopy(self.base_estimator))
        #     self.estimators[i].fit(subsample_X, subsample_y)
