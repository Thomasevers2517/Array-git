import numpy as np
import scipy.linalg as linalg
import scipy.signal as ss

def espirit_DOA(X,d):
    """Perform ESPRIT algorithm to estimate the direction of arrival of multiple sources.
    Parameters
    ----------
    X : numpy.ndarray
        Matrix of data generated
    d : int
        Number of known sources
    
    Returns
    -------
    DOA : list
        List of esitmated angles of arrival of sources
    """
    num_of_sources = d
    delta = X["delta"] # ???How to get spacing between array elements from Data generation functonion
    a_k = X["a_k"] # ???How to get Array response vector from Data generation functonion
    THETA = np.exp(1j*2*np.pi*delta*np.sin(a_k))
    Y = linalg.matmul(X,THETA)
    Z=[X,Y]
    U, S, Vh = linalg.svd(Z,true)
    Theta = [0] * num_of_sources
    DOA = [Theta[i][i] for i in num_of_sources]
    return DOA # return diagonal elements of capital Theta matrix