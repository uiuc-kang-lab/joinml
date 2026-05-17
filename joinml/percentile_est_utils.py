import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def cal_weighted_median(strata_values, strata_probs):
    # clean up None values in each strata
    strata_values_cleaned = []
    strata_probs_cleaned = []
    for stratum_values, stratum_probs in zip(strata_values, strata_probs):
        indices_cleaned = [i for i in range(len(stratum_values)) if stratum_values[i] is not None]
        stratum_values_cleaned = [stratum_values[i] for i in indices_cleaned]
        stratum_probs_cleaned = [stratum_probs[i] for i in indices_cleaned]
        strata_values_cleaned.append(stratum_values_cleaned)
        strata_probs_cleaned.append(stratum_probs_cleaned)
    
    # convert probs to weights
    strata_weights = []
    for stratum_values, stratum_probs in zip(strata_values_cleaned, strata_probs_cleaned):
        if stratum_probs[0] == 1:
            strata_weights.append(stratum_probs)
            continue
        n = len(stratum_values)
        stratum_weights = [1.0 / (n * prob) for prob in stratum_probs]
        strata_weights.append(stratum_weights)

    # combine values and weights
    values = []
    weights = []
    for stratum_values, stratum_weights in zip(strata_values_cleaned, strata_weights):
        values += stratum_values
        weights += stratum_weights

    return weighted_median(values, weights)


def weighted_median(values, weights):
    """ Helper to calculate weighted median efficiently. """
    df = pd.DataFrame({'val': values, 'wt': weights}).sort_values('val')
    df['cum_wt'] = df['wt'].cumsum()
    cutoff = df['wt'].sum() / 2.0
    return df[df['cum_wt'] >= cutoff].iloc[0]['val']

def analytic_variance(D1_values, d2_values, d2_probs):
    """
    Estimates variance using the Influence Function / Sandwich Formula.
    Var ≈ (Sum(1/q_k) / 4n) / (N_total * f(m))^2
    """
    n = len(d2_values)
    
    # 1. Construct the combined dataset to find the point estimate m
    w1 = np.ones(len(D1_values))
    w2 = 1.0 / (n * d2_probs)
    
    all_vals = np.concatenate([D1_values, d2_values])
    all_wts = np.concatenate([w1, w2])
    
    # Estimated Total Population Size
    N_total = np.sum(all_wts)
    
    # Point estimate of the median
    m = weighted_median(all_vals, all_wts)
    
    # 2. Estimate Density f(m) using Gaussian KDE
    # We use the weights to do a weighted KDE
    kde = gaussian_kde(all_vals, weights=all_wts, bw_method='scott')
    f_m = kde(m)[0]
    
    # 3. Calculate Numerator (The Variance of the Score)
    # The formula for the variance contribution of D2 is:
    # Var(Score) ≈ 1/4 * E[ sum(1/q) ] 
    # We estimate sum(1/q) using the sample: sum(1/q^2) / n
    # (Recall: Sample average of g(x)/q(x) estimates population sum of g(x))
    
    # Estimating the sum of inverse probs in the full D2 population
    sum_inv_q_est = np.sum(1.0 / (d2_probs**2)) / n
    
    numerator = (1.0 / 4.0) * sum_inv_q_est
    
    # 4. Final Variance
    # Denominator is squared slope
    denominator = (N_total * f_m) ** 2
    
    return numerator / denominator
