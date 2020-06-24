import numpy as np
import time

class CustomLogisticRegression(object):

    def __init__(self, epochs = 100, batch_size = 32, lr_rate = 0.1, optimizer="momentum", beta1 = 0.9, beta2 = 0.999, weights=[1., 1.]):
        """
            :params:
                epochs - Number of times to iterate over the complete dataset
                batch_size - No. of samples to be used to train the model at a time
                lr_rate - learning rate
                optimizer - types allowed: sgd, momentum, rms
                beta1 - applicable only for momentum
                beta2 - applicable only for rms
                weights - allows weighted loss, [false -ve, false +ve]
        """
        self.weights = weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.__vdw = 0
        self.__sdw = 0

    def __sigmoid(self, z):
        if z.any(axis=1) > 100:
            print(z[z>100])
        return (1.0/(1+np.exp(-z)))
    
    def predict(self, x):
        x = np.array(x)
        z = np.dot(self.coef_, x.T)
        assert z.shape == (1, x.shape[0]), "Z is not what u expected"
        return self.__sigmoid(z)
    
    def score(self, X, y):
        """
            calculates mean accuracy by applying self.predict(X)
            and comparing with y
        """
        y = np.array(y)
        y_pred = self.predict(X)[0]
        y_pred = y_pred >= 0.5
        return(y == y_pred).mean()

    def __getBatches(self, X, y, seed):
        
        idxs = list(range(X.shape[0]))
        np.random.seed(seed)
        np.random.shuffle(idxs)
        
        batches = []

        for start in range(0, X.shape[0], self.batch_size):
            end = X.shape[0] if start + self.batch_size > X.shape[0] else start + self.batch_size 
            
            batches.append((X[idxs[start: end]], y[:,idxs[start: end]]))
        
        return batches
    
    def __momentum(self, dw):
        """
            function to implement sgd with momentum 
        """
        self.__vdw = (self.beta1)*self.__vdw + (1-self.beta1)*dw
        self.coef_ = self.coef_ - self.lr_rate*self.__vdw
    
    def __rmsprop(self, dw):
        """
            function to apply rms prop 
        """
        self.__sdw = (self.beta2)*self.__sdw + (1-self.beta2)*(dw**2)
        self.coef_ = self.coef_ - self.lr_rate * (dw/np.sqrt(self.__sdw + self.epsilon))

    def __sgd(self, dw):
        """
            function to apply pure gradient descent inorder to update coef_ using the passed partial derivaties
        """
        self.coef_ = self.coef_ - self.lr_rate*dw

    def __update(self, y_true, y_pred, x):
        """
            updates coef_ using gradient descent
        """
        diff = y_pred - y_true

        diff[diff < -0.5] = diff[diff < -0.5] *self.weights[0] # false -ve
        diff[diff > 0.5] = diff[diff > 0.5] *self.weights[1] # false +ve

        dcoef_ = (diff.T)*x
        assert dcoef_.shape == (x.shape[0], self.n_features), "Unexpected shape of dcoef_"
        
        dcoef_ = np.mean(dcoef_, axis = 0, keepdims=True)

        if self.optimizer == 'sgd':
            self.__sgd(dcoef_)
        elif self.optimizer == 'momentum':
            self.__momentum(dcoef_)
        elif self.optimizer == 'rms':
            self.__rmsprop(dcoef_)
        
    def cost(self, y_true, y_pred):
        """
            function to cal cost/error
        """
        y_pred[y_pred == 0] = 0.0000001
        y_pred[y_pred == 1] = 0.9999999
        
        return np.mean( -y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred))

    def fit(self, X, y, verbose = 2):
        """
            X - 2D array or a DataFrame of features,
                each row being a sample observation
            Y - 1D array or a Series consisting of target variable

            verbose - (1, 2) 
                        1 - print the progress over each epoch
                        2 - just the final cost
        """
        X = np.array(X)
        y = np.array(y)
        assert X.ndim == 2, "Invalid shape of X"
        
        if y.ndim == 1:
            y = y.reshape(1,-1)

        self.n_features = X.shape[1]
        # self.coef_ = np.random.randn(1, self.n_features)
        self.coef_ = np.zeros((1, self.n_features))
        seed = 0
        for curr_epoch in range(1, self.epochs+1):
            cost = 0
            seed += 1
            self.__vdw = np.zeros_like(self.coef_)
            self.__sdw = np.zeros_like(self.coef_)

            for (x_batch, y_batch) in self.__getBatches(X, y, seed):
                
                y_batch_pred = self.predict(x_batch)
                self.__update(y_batch, y_batch_pred, x_batch)
                cost += self.cost(y_batch, y_batch_pred)
            
            if verbose == 1:
                print("Epoch: {} Cost = {:,.2f}".format(curr_epoch, cost))
                time.sleep(0.25)
            
        if verbose == 2:
            print("Cost after final epoch: {:.2f}".format(cost))
    
    def __str__(self):
        return "CustomLogisticRegression("+', '.join([f'{key}={val}' for key,val in self.__dict__.items()])+")"