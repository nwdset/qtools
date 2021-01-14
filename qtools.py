import numpy as np
import pandas as pd
from numba import njit

freq_adj = {
    'D': 15.874507866387,
    'W': 7.2111025509279,
    'M': 3.4641016151377,
}

@njit(cache=True)
def sr(rets :np.ndarray, freq='D') -> np.float64:
    """ Computes Sharpe Ratio for a given array

    Parameters
    ----------
    rets : np.ndarray, shape (T, n)
        T is observations and n is number of assets

    """
    
    # Sharpe ratio
    sr = np.mean(rets, axis=1)/np.std(rets, axis=1)
    
    return sr*freq_adj[freq]

@njit(cache=True)
def eccov(data: np.ndarray) -> np.ndarray:
    """ Denoising cov matrix via eigenvalue clipping

    Parameters
    ----------
    data : np.ndarray
        Data to be used for covariance estimation

    Returns
    -------
    cov : np.ndarray
        Denoised covariance matrix
    """

    # parameters and raw data
    T          = data.shape[0]
    n          = data.shape[1]
    lmax       = (1+sqrt(n/T))**2
    corr       = np.corrcoef(data.T)
    stds       = np.sqrt(np.diag(np.cov(data.T)))
    vals, vecs = np.linalg.eig(corr)
    vals, vecs = np.real(vals), np.real(vecs)
    # denoising matrix
    vals[vals<lmax] = np.mean(vals[vals<lmax])
    corr            = vecs @ np.diag(vals) @ vecs.T
    cov             = np.diag(stds) @ corr @ np.diag(stds)

    return cov

@njit(cache=True)
def mkri(dt: int, T: int) -> np.ndarray:
    """Creates rolling indices for numpy arrays 

    Parameters
    ----------
    dt : int 
        Rolling window of timeseries

    Returns
    -------
    index : np.ndarray, shape (T-1, dt, n)
    """

    rows  = np.expand_dims(np.arange(dt), 0)
    cols  = np.expand_dims(np.arange(T-dt+1), 1)
    index = rows + cols

    return index

def RollingOLS(endog: pd.DataFrame, exog: pd.DataFrame, window=2) -> pd.DataFrame:
    """ Compute a Rolling OLS for DataFrames

    Parameters
    ----------
    endog : pd.DataFrame
        Endogenous variable

    exog : pd.DataFrame
        Exogenous variable, with a constant of 1s

    Returns
    -------
    params : pd.DataFrame 
        Rolling OLS parameters

    """
    
    T      = endog.shape[0]
    index  = mkri(window, T)
    ry     = endog.values[index].reshape(-1, window, 1)
    rX     = exog.values[index]
    rXt    = rX.transpose(0, 2, 1)
    params = ((np.linalg.inv(rXt @ rX) @ rXt) @ ry).reshape(-1, 2)
    params = pd.DataFrame(params, endog.index[window-1:], exog.columns)
    return params
