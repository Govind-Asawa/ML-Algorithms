import numpy as np

class CustomKNNRegressor(object):

    EUCLIDEAN = 0
    MANHATTAN = 1
    COSINE = 3

    def __init__(self, k, dist = EUCLIDEAN, random_state=None):
        """
            :params:
                k - number of neighbours
                dist - type of distance metric to use
        """
        self.k = k
        self.random_state = random_state
        self.dist = dist
        self.__distances = {
            CustomKNNRegressor.EUCLIDEAN : self.__euclidean,
            CustomKNNRegressor.MANHATTAN : self.__manhattan,
            CustomKNNRegressor.COSINE : self.__cosine
        }
    
    def __euclidean(self, a, b):
        """
            sqrt((a1-b1)^2 + (a2-b2)^2 + ... +(an-bn)^2)
        """
        return np.sqrt(np.sum((a-b)**2)) 
    
    def __manhattan(self, a, b):
        """
            |a1-b1| + |a2-b2| + .... +|an-bn|
        """
        return np.sum(np.abs(a-b))

    def __cosine(self, a, b):
        """
              a.b
          ----------
          ||a|| ||b||

        """
        return np.sum(a*b)/( np.sqrt(np.sum(a**2)) + np.sqrt(np.sum(b**2)) )

    def __getNeighbours(self, test_sample):
        """
            function to calculate the specified dist 
            between the passed test_sample and all the training samples
            and returns the target values of k nearest neighbours
        """
        neighbours = []
        for train_sample_x, train_sample_y  in zip(self.X, self.y):
            neighbours.append( (train_sample_y, self.__distances[self.dist](train_sample_x, test_sample) ))
        
        neighbours.sort(key = lambda item:item[1])
        return np.array([sample_y for sample_y, _ in neighbours[:self.k]])

    def predict(self, test_X):
        
        predictions = []
        for test_sample in test_X:
            knearest_neighbours = self.__getNeighbours(test_sample)
            predictions.append(np.mean(knearest_neighbours))
        
        return np.array(predictions)

    def fit(self, X, y):
        """
            X - 2D array or a DataFrame of features,
                each row being a sample observation
            Y - 1D array or a Series consisting of target variable
        """
        X = np.array(X)
        y = np.array(y)
        assert X.ndim == 2, "incompatible size of X"
        assert y.ndim == 1, "incompatible size of y"
        
        self.X = X
        self.y = y