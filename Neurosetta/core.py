import numpy as np
import vaex as vx
import graph_tool.all as gt
import os

### Classes

class Neuron_Tree():
    """
    Core Neuron tree class
    """

    __slots__ = ["name","node_table","graph","summary_table"]

    def __init__(self, node_table, name = None, graph = None):

        # type checks

        # add name
        if name is None:
            self.name = 'Neuron'
        else:
            assert isinstance(name,str), "Provided neuron name is not a string"
            self.name = name
        
        # add node table
        self.node_table = node_table

        # add / generate graph
        if graph is None:
            self.graph = graph_from_vaex(node_table)
        else:
            assert isinstance(graph,gt.Graph), "Provided graph is not a Graph-Tool graph"
            self.graph = graph

        self.summary_table = None
    
    # Generator functions
    def summary(self):
        if 'distance' in self.node_table.column_names:
            summary = vx.from_scalars(name = self.name,
                                        nodes = _count_nodes(self),
                                        branches = _count_branch_nodes(self),
                                        ends = _count_end_nodes(self),
                                        segments = _count_segs(self),
                                        cable_length = _total_cable_length(self)
                                        )
        else:
            summary = vx.from_scalars(name = self.name,
                                        nodes = _count_nodes(self),
                                        branches = _count_branch_nodes(self),
                                        ends = _count_end_nodes(self),
                                        segments = _count_segs(self),
                                        )
        self.summary_table = summary
        return summary
    
    # add attribute functions
    def classify_nodes(self, overwrite = True):

        return _classify_nodes(self, overwrite = overwrite)
    
    def add_distance(self):

        return _add_distances(self)
   
    # metric functions
    def count_nodes(self):
        return _count_nodes(self)

    def count_branch_nodes(self):
        return _count_branch_nodes(self)

    def get_branch_ids(self):
        return _get_branch_nodes(self)

    def count_end_nodes(self):
        return _count_end_nodes(self)

    def get_end_ids(self):
        return _get_end_nodes(self)

    def count_segs(self):
        return _count_segs(self)

    def get_root_id(self):
        return _get_root_id(self)

    def total_cable(self):
        if 'distance' not in self.node_table.column_names:
            self = _add_distances(self)
        return _total_cable_length(self)
    
    def get_coordinates(self,cols = None):
        if cols is None:
            return _get_cols(self,['x','y','z'])
        else:
            return _get_cols(self,cols)
        
### reading data from file

def read_swc(file_path, add_distances = True, classify_nodes = True):
    """
    Generate neuron from swc
    """
    # neuron class inputs
    df = vaex_from_swc(file_path)
    
    g = graph_from_vaex(df)

    name = os.path.splitext(os.path.basename(file_path))[0]
    # generate neuron
    N = Neuron_Tree(name = name,node_table = df, graph=g)

    if classify_nodes == True:
        N.classify_nodes()
    if add_distances == True:
        N.add_distance()

    
    
    return N

def vaex_from_swc(file_path):
    """
    Read in swc file to a vaex DataFrame.

    Paremeters
    ----------

    file_path:          str
        file path string

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
    
    return df

### Generating bits from memory

def graph_from_vaex(df):
    """
    Creates a graph-tool graph from vaex data frame neuron representation
    """
    # edge array
    edges = df['parent_id','node_id'].values.astype(int)

    # cut root from edges
    edges = edges[np.where(edges[:,0] != np.setdiff1d(edges[:,0],edges[:,1])[0])]

    g = gt.Graph(edges, hashed = True, hash_type = 'int')

    # add some attributes which should be in all swc files
    # initialise vertex ID, coordinates, and radius
    vprop_rad = g.new_vertex_property('double')
    vprop_coord = g.new_vertex_property('vector<double>')

    vprop_rad.a = df.radius.values
    vprop_coord.set_2d_array(df['x','y','z'].values.T)

    # add to graph
    g.vp['radius'] = vprop_rad
    g.vp['coordinates'] = vprop_coord

    # add total degree property map
    g.vp['degree_total'] = g.degree_property_map("total")

    return g

### Adding attributes

def _classify_nodes(N, overwrite = True):
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
        N.node_table['type'] = N.node_table.func.where(N.node_table.type != np.int32(0), np.int32(0), N.node_table.type)
        

    # types
    out_deg = N.graph.get_out_degrees(N.graph.get_vertices())
    in_deg = N.graph.get_in_degrees(N.graph.get_vertices())
    ends = np.where(out_deg == 0)
    branches = np.where(out_deg > 1)
    root = np.where(in_deg == 0)
    node_types = np.zeros_like(N.graph.get_vertices())
    node_types[ends] = 6
    node_types[branches] = 5
    node_types[root] = 1
    N.node_table['type'] = node_types

    # initialise type property
    vprop_type = N.graph.new_vertex_property('int')
    # populate
    vprop_type.a = node_types
    # add to graph
    N.graph.vp['type'] = vprop_type
    
    return N

def _add_distances(N):
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

    # distances
    distances = np.zeros(len(N.node_table))
    nodes = N.node_table['node_id'].values
    r = N.get_root_id()
    
    for i in N.graph.iter_edges():
        dist = np.linalg.norm(N.graph.vp['coordinates'][i[0]].a - N.graph.vp['coordinates'][i[1]].a)
        distances[np.where(nodes == N.graph.vp['ids'][i[1]])[0]] = dist

    N.node_table['distance'] = distances

    # add distance property to edges - need to remove 0
    distances = distances[np.where(nodes != r)]

    eprop_dist = N.graph.new_edge_property('double')
    eprop_dist.a = distances
    N.graph.ep['distance'] = eprop_dist
    
    return N

### metrics

def _count_nodes(N):
    """
    Returns the number of nodes in the neuron
    """
    return len(N.node_table)

def _count_branch_nodes(N):
    """
    Returns the number of branch nodes
    """
    return len(N.node_table[N.node_table.type == 5])

def _get_branch_nodes(N):
    """
    returns the node id of branch nodes
    """
    return N.node_table[N.node_table.type == 5]['node_id'].values

def _count_end_nodes(N):
    """
    returns the number of end nodes
    """
    return len(N.node_table[N.node_table.type == 6])

def _get_end_nodes(N):
    """
    returns the node ids of end nodes
    """
    return N.node_table[N.node_table.type == 6]['node_id'].values

def _count_segs(N):
    """
    returns the number of segemnts within the neuron
    """
    return _count_branch_nodes(N) + _count_end_nodes(N)

def _get_root_id(N):
    """ 
    returns the node id of the root
    """
    return N.node_table[N.node_table.type == 1]['node_id'].values[0]

def _total_cable_length(N):
    """
    Returns the total cable length of the neuron
    """
    if 'distance' not in N.node_table.get_column_names():
        print('Distances not available - add distances')
    return np.sum(N.node_table['distance'].values)

def _get_cols(N,cols = None):
    """
    returns np.array of specified columns in node table
    """
    if cols is None:
        raise AttributeError('please provide a list of columns you wish to return')
    else:
        return N.node_table[cols].values



### Temp functions which need to be fleshed out 

def Neuron_list(path):
    """
    Return a list of neuron objects
    """
    Ns = []