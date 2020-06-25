import numpy as np
import time

class CustomLinearRegression(object):

    def __init__(self, batch_size = 32, epochs = 50, lr_rate = 0.1, optimizer = 'sgd'):
        """
        :params:
            batch_size - No. of samples to be used to train the model at a time
            epochs - Number of times to iterate over the complete dataset
            lr_rate - learning rate
            optimizer - types allowed: sgd
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_rate = lr_rate
        self.optimizer = optimizer
    
    def __getBatches(self, X, y, seed):
        """
            function to generate batches out of X and y with each having <= self.batch_size samples

            :returns:
                list of batches
        """
        np.random.seed(seed)
        idxs = list(range(X.shape[0]))
        np.random.shuffle(idxs)

        batches = []
        for i in range(0, X.shape[0], self.batch_size):
            batches.append( (X[idxs[i:i+self.batch_size], :], y[:, idxs[i:i+self.batch_size]]) )
        
        return batches
    
    def __sgd(self, dw, db):
        """
            function that applies stochastic gradient descent
        """
        self.coef_ = self.coef_ - self.lr_rate*dw
        self.intercept_ = self.intercept_ - self.lr_rate*db

    def __update(self, y, pred_y, X):
        """
            function that updates current coef_ (weihghts) based on the error
        """
        dcoef_ = np.mean( (pred_y - y).T*X, axis=0, keepdims=True )
        dintercept_ = np.mean(pred_y - y)
        assert dcoef_.shape == self.coef_.shape, "invalid shape of dcoef"

        if self.optimizer == 'sgd':
            self.__sgd(dcoef_, dintercept_)

    def cost(self, y, pred_y):
        """
            :return:
                MSE (mean squared error)
        """
        return np.mean((y-pred_y)**2)

    def predict(self, X):
        return np.dot(self.coef_, X.T) + self.intercept_

    def fit(self, X, y, verbose = 3):
        """
        :params:
            X - 2D array or a DataFrame of shape (m, n), where
                m is the number of samples and n is the number of features
            Y - 1D array or Series of shape (1, m), consisting of target variable
            verbose - int, 1, 2 or 3
                    1 - print the progress over each epoch
                    2 - just the final cost
                    3 - no ack
        """
        
        X = np.array(X)
        y = np.array(y)

        assert X.ndim == 2, "incompatible shape of X"
        assert y.ndim == 1, "incompatible shape of y"

        y = y.reshape(1, -1)

        self.coef_ = np.zeros((1, X.shape[1]))
        self.intercept_ = 0
        seed = 0
        for epoch in range(1, self.epochs+1):
            seed += 1
            cost = 0
            for batch_X, batch_y in self.__getBatches(X, y, seed):

                pred_y = self.predict(batch_X)
                assert pred_y.shape == batch_y.shape, "unexpected shape of pred_y"
                # print(pred_y)
                self.__update(batch_y, pred_y, batch_X)
                cost += self.cost(batch_y, pred_y)
            
            if verbose == 1:
                time.sleep(0.25)
                print(f"cost after {epoch} epochs: {cost:.3f}")
        
        if verbose == 2:
            print(f"cost after final epochs: {cost:.3f}")
        
        return self
    
    def __str__(self):
        return "CustomLinearRegression("+', '.join([f'{key}={val}' for key,val in self.__dict__.items()])+")"