import numpy as np
import vaex as vx
import graph_tool.all as gt


    

def read_swc(file_path, add_distances = False, add_types = True):
    """
    Read in swc file to a vaex DataFrame.

    Paremeters
    ----------

    file_path:          str
        file path string

    add_distances:      bool
        If True (default), distance from child to parent will be calculated, add paren x,y,z, and distance columns to the returned data frame

    add_types:          bool
        if True (default) existing node classification in the swc file will be overwritten, and nodes will be re-classified as root = 1, branch = 5, leaf = 6, other = 0

    Returns
    -------

    df                  vaex.DataFrame
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

    if add_types == True:
        df = classify_nodes(df)

    if add_distances == True:
        df = get_distances(df)
    


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
    if df[df.node_id == r].parent_id.values != -1:
        df['parent_id'] = df.func.where(df.node_id.isin(r),-1,df.parent_id)
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
    # root parent
    rp = df[df.parent_id.isin(np.setdiff1d(df.parent_id.values,df.node_id.values))].parent_id.values[0]
    # add parent coordinates
    df['px'] = np.array([df[df.node_id == p]['x'].values[0] if p!= rp else df[df.node_id == c]['x'].values[0] for c,p in df['node_id','parent_id'].values])
    df['py'] = np.array([df[df.node_id == p]['y'].values[0] if p!= rp else df[df.node_id == c]['y'].values[0] for c,p in df['node_id','parent_id'].values])
    df['pz'] = np.array([df[df.node_id == p]['z'].values[0] if p!= rp else df[df.node_id == c]['z'].values[0] for c,p in df['node_id','parent_id'].values])
    # add distances
    df['distance'] = np.linalg.norm((df.x - df.px,df.y - df.py,df.z - df.pz))  
    
    return df

def count_nodes(df):
    """
    Returns the number of nodes in the neuron
    """
    return len(df)

def count_branch_nodes(df):
    """
    Returns the number of branch nodes
    """
    return len(df[df.type == 5])

def get_branch_nodes(df):
    """
    returns the node id of branch nodes
    """
    return df[df.type == 5]['node_id'].values

def count_end_nodes(df):
    """
    returns the number of end nodes
    """
    return len(df[df.type == 6])

def get_end_ndoes(df):
    """
    returns the node ids of end nodes
    """
    return df[df.type == 6]['node_id'].values

def count_segments(df):
    """
    returns the number of segemnts within the neuron
    """
    return count_branch_nodes(df) + count_end_nodes(df)

def total_cable_length(df):
    """
    Returns the total cable length of the neuron
    """
    if 'distance' not in df.get_column_names():
        get_distances(df)
    return np.sum(df['distance'].values)

def create_graph(df, include_distance = False):
    """
    Creates a graph-tool graph from vaex data frame neuron representation
    """
    if include_distance:
        if 'distance' in df.column_names:
            # generate graph from edge list with distance as an edge property
            g = gt.Graph([(p-1,c-1,d) for p,c,d in df['parent_id','node_id','distance'].values if p != -1], eprops = [('distance','double')])
        else:
            df = get_distances(df)
            g = gt.Graph([(p-1,c-1,d) for p,c,d in df['parent_id','node_id','distance'].values if p != -1], eprops = [('distance','double')])
    else:
        g = gt.Graph([(p-1,c-1) for p,c in df['parent_id','node_id'].values if p != -1])


    # initialise vertex properties
    #radius
    vprop_rad = g.new_vertex_property('double')
    # id
    vprop_id = g.new_vertex_property('int')
    # node type
    vprop_type = g.new_vertex_property('int')

    # populate properties
    ids = g.get_vertices() + 1
    vprop_id.a = ids
    vprop_rad.a = df[df.node_id.isin(ids)].radius.values
    vprop_type.a = df[df.node_id.isin(ids)].type.values
    vprop_coord = g.new_vertex_property('vector<double>')
    vprop_coord.set_2d_array(df[df.node_id.isin(ids)]['x','y','z'].values.T)

    # add them to graph
    g.vp['radius'] = vprop_rad
    g.vp['ID'] = vprop_id
    g.vp['type'] = vprop_type
    g.vp['coordinates'] = vprop_coord

    return g