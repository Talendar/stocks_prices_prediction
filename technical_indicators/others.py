import numpy as np
import pandas as pd


def stochastic_oscillator(data, n=14, nans=-1):
  """ Computes the stochastic oscillator (%K). """
  so = np.full(len(data), nans, dtype=np.float)
  for i in range(n - 1, len(data["close"])):
    past_n = data.iloc[(i - n + 1):(i + 1)]
    curr, low, high = data["close"].iloc[i], np.min(past_n["low"]), np.max(past_n["high"])
    so[i] = 100 * (curr - low) / (high - low)
  
  return pd.DataFrame(so, index=data.index, columns=[f"%K{n}"])
