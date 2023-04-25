import numpy as np
from scipy.stats import ttest_ind

def boot_welchTtest(s1,s2,bootstraps):
    """
    Bootstrapped welches two sample t-test
    """
    # compute t-statistic of data and store
    obs_t = ttest_ind(s1,s2,equal_var = False)[0]
    # change data so that null hypothesis is true - in this case, subtract group
    # mean from each group, and add the global mean
    null_1 = (s1 - np.mean(s1)) + np.mean(list(s1) + list(s2))
    null_2 = (s2 - np.mean(s2)) + np.mean(list(s1) + list(s2))

    # initialise t-values
    b_ts = np.zeros(bootstraps)
    # bootstrapping
    for i in range(bootstraps):
        bs1 = np.random.choice(null_1,len(null_1),replace = True)
        bs2 = np.random.choice(null_2,len(null_1),replace = True)
        b_ts[i] = ttest_ind(bs1,bs2,equal_var = False)[0]

    p = (len(b_ts[b_ts>=obs_t]) + 1) / (bootstraps + 1)
    
    return obs_t, p