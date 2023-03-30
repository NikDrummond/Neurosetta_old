import numpy as np
import vaex as vx

def read_swc(file_path, distances = True, classify_nodes = True):
    """
    Read in swc file to a vaex DataFrame.

    Paremeters
    ----------

    file_path:      str
        file path string

    distances:      bool
        If True (default), distance from child to parent will be calculated, add paren x,y,z, and distance columns to the returned data frame

    classify_nodes: bool
        if True (default) existing node classification in the swc file will be overwritten, and nodes will be re-classified as root = 1, branch = 5, leaf = 6, other = 0

    Returns
    -------

    df              vaex.DataFrame
        
    

    """
    df = vx.read_csv(file_path, 
              names = ['node_id','type','x','y','z','radius','parent_id'],
              comment = '#',
              engine = 'c',
              delim_whitespace = True,
              dtype = {'node_id':np.int32,
                        'type':np.int32,
                        'x':np.float64,
                        'y':np.float64,
                        'z':np.float64,
                        'radius':np.float64,
                        'parent_id':np.int32})
    if not np.unique(df.node_id.values).size == len(df.node_id.values):
        raise AttributeError('Duplicate Node Ids found')

    if distances == True:
        df = get_distances(df)
    
    if classify_nodes == True:
        df = classify_nodes(df)

    return df

def classify_nodes(df, overwrite = True):
    """
    Classifies branch, end and root nodes in diven DataFrame. Classifies the root as 1, branches as 5, and end nodes as 6. 
    everything else is set to 0 if overwrite == True

    Parameters
    ----------

    df:         vaex.DataFrame
        vaex data frame representing given neuron, as returned by read_swc

    overwrite:  bool
        if True(default) will overwrite existinc type column

    Returns
    -------

    df          vaex.DataFrame
        Original dataframe with updated node types
    """
    if overwrite == True:
        df['type'] = df.func.where(df.type != np.int32(0), np.int32(0), df.type)

    # ends
    e = df[df.node_id.isin(np.setdiff1d(df.node_id.values,df.parent_id.values))].node_id.values
    df['type']= df.func.where(df.node_id.isin(e),6,df.type)

    # root
    r = df[df.parent_id.isin(np.setdiff1d(df.parent_id.values,df.node_id.values))].node_id.values
    # if the parent of the root node is not -1, update it
    if r != -1:
        df['parent_id'] = df.func.where(df.parent_id.isin(r),-1,df.parent_id)
    df['type']= df.func.where(df.node_id.isin(r),1,df.type)

    # branches
    b = np.sort(df.parent_id.values)
    b = np.unique(b[:-1][b[1:] == b[:-1]])
    df['type']= df.func.where(df.node_id.isin(b),5,df.type)
    
    return df

def get_distances(df):
    """
    Add child to parent distance to Data Frame (in the same units as the coordinate system). Also adds [px,py,pz] columns, which are the parent coordinates for each node.
    if the node is the root, [px,py,pz] = [x,y,z]

    Parameters
    ----------

    df:     vaex.DataFrame
        Data frame of Neuron

    Returns
    -------

    df:     vaex.DataFrame
        original DataFrame with added node distances and parent coordinates.

    """
    # root
    r = df[df.parent_id.isin(np.setdiff1d(df.parent_id.values,df.node_id.values))].node_id.values
    # add parent coordinates

    df['px'] = np.array([df[df.node_id == p]['x'].values[0] if p!= r else df[df.node_id == c]['x'].values[0] for c,p in df['node_id','parent_id'].values])
    df['py'] = np.array([df[df.node_id == p]['y'].values[0] if p!= r else df[df.node_id == c]['y'].values[0] for c,p in df['node_id','parent_id'].values])
    df['pz'] = np.array([df[df.node_id == p]['z'].values[0] if p!= r else df[df.node_id == c]['z'].values[0] for c,p in df['node_id','parent_id'].values])
    # add distances
    df['distance'] = np.linalg.norm((df.x - df.px,df.y - df.py,df.z - df.pz))  
    
    return df

def count_nodes(df):
    """
    
    """
    return len(df)

def count_branch_nodes(df):
    """
    
    """
    return len(df[df.type == 5])

def get_branch_nodes(df):
    """
    
    """
    return df[df.type == 5]['node_id'].values

def count_end_nodes(df):
    """
    
    """
    return len(df[df.type == 6])

def get_end_ndoes(df):
    """
    
    """
    return df[df.type == 6]['node_id'].values

def count_segments(df):
    """
    
    """
    return count_branch_nodes(df) + count_end_nodes(df)

def total_cable_length(df):
    """
    
    """
    if 'distance' not in df.get_column_names():
        get_distances(df)
    return sum(df['distance'])

