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

def forest_summary(df):
    """
    Given df of multiple neurons, produce summary table
    """
    # df_forest summary
    fs = df.groupby(by = 'N_id').agg({'nodes':'count',
                                        'branches':vx.agg.count('type',selection=('type == 5')),
                                        'ends':vx.agg.count('type',selection=('type == 6')),
                                        'cable_length': vx.agg.sum('path_length')}).copy()
    fs['segments'] = fs['branches'] + fs['ends']

    return fs