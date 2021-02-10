import numpy as np

# Compute anomaly scores:
class MahalaDistance():
    """
    Compute Mahalanobis distribution of a dataset with 
    respect to a normal distribution defined by maximum likelihood 
    parameter estimates. 
    
    Parameter estimates can be passed directly via optional keyword 
    arguments, or they will be fit from the data
    
    """
    def __init__(self, mu=None, cov=None):
        """
        :param data: multi-variate data.
        :param cov_inv: Optional. Inverse of covariance parameter 
            estimate.
        :param mu: Optional. Mean parameter estimate.
        :return: A tuple containing (distances, inv_cov, mu). 
        """
        
        self.mu = mu
        self.cov = cov
        self.cov_inv = None
        
        if cov is not None:
            self.cov_inv = np.linalg.inv(cov)
    
    def fit(self, data):
        """
        Compute normal distribution maximum likelihood parameter estimates.
        :param data: Data from which to calculate estimates. 
        """
        self.cov = np.cov(data, rowvar=False)
        self.cov_inv = np.linalg.inv(self.cov)
        self.mu = np.mean(data, axis=0)
        
        return self.mu, self.cov
            
    
    def measure(self, data):
        """
        Measure Mahalanobis distribution based on N(mu, sigma) distribution.
        :param data: The data to measure. 
        """
        dist = []
        for x in data:
            md = self._compute_mahala(x)
            dist.append(md)
            
        return np.array(dist).squeeze()
    
    def fit_measure(self, data):
        """
        Estimate mu and sigma of N(mu, sigma) and then measure data 
        distances.
        """
        self.fit(data)
        dists = self.measure(data)
        return dists, self.mu, self.cov

    def _compute_mahala(self, observation):
        '''
        Defines a function that calculates the Mahalanobis distance
        from a single observation to data center of mass of a distribution.
        '''
        sample = np.atleast_2d(observation)
        mu = np.atleast_2d(self.mu)
        return np.sqrt((sample - self.mu) @ self.cov_inv @ (sample - self.mu).T)