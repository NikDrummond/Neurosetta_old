import numpy as np
import vaex as vx

def read_swc(file_path):
    """
    
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
    # add parent coordinates

    df['px'] = np.array([df[df.node_id == p]['x'].values[0] if p!= -1 else df[df.node_id == c]['x'].values[0] for c,p in df['node_id','parent_id'].values])
    df['py'] = np.array([df[df.node_id == p]['y'].values[0] if p!= -1 else df[df.node_id == c]['y'].values[0] for c,p in df['node_id','parent_id'].values])
    df['pz'] = np.array([df[df.node_id == p]['z'].values[0] if p!= -1 else df[df.node_id == c]['z'].values[0] for c,p in df['node_id','parent_id'].values])
    # add disnatces
    df['distance'] = np.linalg.norm((df.x - df.px,df.y - df.py,df.z - df.pz))  
    
    return df

def classify_nodes(df, overwrite = True):
    """

    """
    if overwrite == True:
        df['type'] = df.func.where(df.type != np.int32(0), np.int32(0), df.type)

    # ends
    e = df[df.node_id.isin(np.setdiff1d(df.node_id.values,df.parent_id.values))].node_id.values
    df['type']= df.func.where(df.node_id.isin(e),6,df.type)

    # root
    r = df[df.parent_id.isin(np.setdiff1d(df.parent_id.values,df.node_id.values))].node_id.values
    df['type']= df.func.where(df.node_id.isin(r),1,df.type)

    # branches
    b = np.sort(df.parent_id.values)
    b = np.unique(b[:-1][b[1:] == b[:-1]])
    df['type']= df.func.where(df.node_id.isin(b),5,df.type)
    
    return df

def get_distances(df):
    """
    
    """
    dist = []
    for i in df['node_id','parent_id'].values:
        if i[1] == -1:
            dist.append(np.linalg.norm(df[df.node_id == i[0]]['x','y','z'].values - df[df.node_id == i[0]]['x','y','z'].values))
        else:
            dist.append(np.linalg.norm(df[df.node_id == i[0]]['x','y','z'].values - df[df.node_id == i[1]]['x','y','z'].values))
    df['distance'] = np.array(dist).astype(np.float64)

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
    return sum(df['diustance'])

