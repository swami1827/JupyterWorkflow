import pandas as pd
def drawdown(return_series:pd.Series):
    """
    Takes in a time series of asset returns
    computes and returns a dataframe consisting of:
    wealth index
    previous peaks
    percentage drawdowns
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdown= (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "drawdowns": drawdown
    })

def get_ffme_returns():
    """
    Load the Fama French data set for the returns of top and botttom deciles by market cap
    """
    me_m=pd.read_csv("info/Imp_data.csv",
           header=0,index_col=0,parse_dates=True,na_values=-99.99
           )  #monthly returns based on market equity.
    rets= me_m[['Lo 10','Hi 10']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m")
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    
    """
    hfi=pd.read_csv("info/edhec-hedgefundindices.csv",
                   header=0,index_col=0,parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def semideviation(r):
    """
    Computes semideviation/aka negative semideviation of r.
    r must be a series or DataFrame
    """
    is_negative= r<0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to skipy.stats.skew()(direct method)
    Computes the skewness of the given DataFrame or supplied series.
    Returns float or series.
    """
    demeaned_r= r-r.mean()
    # use the popular standard deviation, so setting ddof=0(degree of freedom)
    sigma_r=r.std(ddof=0)
    exp= (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to skipy.stats.kurtosis()(direct method)
    Computes the kurtosis of the given DataFrame or supplied series.
    Returns float or series.
    """
    demeaned_r= r-r.mean()
    # use the popular standard deviation, so setting ddof=0(degree of freedom)
    sigma_r=r.std(ddof=0)
    exp= (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_normal(r,level=0.01):
    """
    Aplies Jarque bera test to verify if a series is normal or not. 
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level

import numpy as np
def var_historic(r, level=5):
    """
    VaR historic
    """
    if isinstance(r, pd.DataFrame):
       return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
       return -np.percentile(r,level) #since we want positive VaR
    else:
        raise TypeError("Expected r to be series or DataFrame")
        

from scipy.stats import norm
def var_gaussian(r, level=5,modified=False):
    """
    Returns the parametric gaussian VaR of a series or DataFrame
    If modified is "True", modified VaR is returned, using the Cornish-Fisher modification.
    """
    #Compute z-score assuming it is gaussian.
    z=norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis.
        s=skewness(r)
        k=kurtosis(r)
        z=(z+
              (z**2-1)*s/6+
              (z**3-3*z)*(k-3)/24-
               (2*z**3-5*z)*(s**2)/36
          )
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r,level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r,pd.Series):
        is_beyond= r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected r to be a series or DataFrame")