from .core import *
import vaex as vx

def forest_df(neurons):
    """
    Given a list of neuron objects, return a single vx data frame
    """

    dfs = []
    for n in neurons:
        df_tmp = n.node_table.copy()
        name = n.name
        df_tmp['N_id'] = np.array([name for i in range(len(df_tmp))])
        dfs.append(df_tmp)

    return vx.concat(dfs)
