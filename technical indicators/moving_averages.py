
def sma(data, n, nans=-1):
    """ Computes the simple MA for each row of the input. 
    
    :param data: pandas dataframe.
    :param n: MA period.
    :return: a new dataframe with the MA values.
    """
    if type(data) == pd.Series:
        data = data.to_frame()

    ma = data.rolling(window=n).mean().fillna(nans)
    ma.columns = [f"{c}_sma{n}" for c in data.columns]
    return ma


def ema(data, n, nans=-1):
    """ Computes the exponential MA for each row of the input. 
    
    :param data: pandas dataframe.
    :param n: MA period.
    :return: a new dataframe with the MA values.
    """
    if type(data) == pd.Series:
        data = data.to_frame()

    ma = data.ewm(span=n, adjust=False).mean().fillna(nans)
    ma.columns = [f"{c}_ema{n}" for c in data.columns]
    return ma


def macd(data, nans=-1):
    """ Computes the MA convergence divergence for each row of the input. 
    
    :param data: pandas dataframe.
    :param n: MA period.
    :return: a new dataframe with the MA values.
    """
    if type(data) == pd.Series:
        data = data.to_frame()

    ma = ema(data, 12, nans).values - ema(data, 26, nans).values
    ma = pd.DataFrame(ma, index=data.index, columns=[f"{c}_macd" for c in data.columns])
    return ma
