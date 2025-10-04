import numpy as np
from scipy import stats

def create_boundaries(X_train: np.ndarray, bins:int) -> tuple:
    '''
    Create equal-probability bin boundaries using Gaussian distribution.
    
    Input:
        - X_train: np.ndarray
        - bins : int
    
    Output:
        - tuple of:
            - mean : float
            - std : float
            - boundaries : np.ndarray
    '''
    L = []
    for x in X_train:
        L.extend(list(x[0]))
        
    probabilities = np.linspace(0, 1, bins + 1)
    
    mean = np.array(L).mean()
    std = np.array(L).std()
    
    # Use the inverse CDF (percent point function) to find the boundaries
    boundaries = stats.norm.ppf(probabilities, loc=mean, scale=std)
    
    return mean, std, boundaries   


def translate_timeseries(dataset: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Function that return the indices of the bins to which each value in input array belongs

    Input:
        - dataset : np.ndarray
        - boundaries : np.ndarray
        
    Output:
        - np.ndarray
    """
    
    result = np.digitize(dataset, boundaries[1:-1], right=False) + 1
    
    return result