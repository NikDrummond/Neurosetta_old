from .core import *
import graph_tool.all as gt

def classify_segments(N):
    """
    Given a neuron, add segment IDs to graph
    """
    g = N.graph.copy()

    # initialise segement and seen children properties
    segment = g.new_vp('int')
    seen_children = g.new_vp('int')
    edge_segment = g.new_ep('int')

    segment_count = 1

    for e in gt.bfs_iterator(g,g.vertex(0)):

        # if we are at the root (total degree = 1)
        if g.vp['degree_total'][e.source()] == 1:
            segment[e.source()] = segment_count
            segment[e.target()] = segment_count
            seen_children[e.source()] = 1

            edge_segment[e] = segment[e.target()]

        # or otherwise this is a transitive node and adopts parent segment
        elif g.vp['degree_total'][e.source()] == 2:
            segment[e.target()] = segment[e.source()]
            seen_children[e.target()] = 1
            edge_segment[e] = segment[e.target()]

        elif g.vp['degree_total'][e.source()] > 2:

            # we are adding new segment!
            segment_count += 1

            segment[e.target()] = segment_count
            seen_children[e.source()] = seen_children[e.source()] + 1 

            edge_segment[e] = segment[e.target()]

    g.ep['segment'] = edge_segment

    N.graph = g

    return N

def simplify_N(N):
    # given neurons (N) return simplified version

    # generating an edge list for a simplified graph - using a depth first search

    # get the start points of all segments
    seg_starts = list(N.get_branch_ids())
    seg_starts.append(N.get_root_id())
    # for counting which row we are on 
    i = 0
    # initialise what will become the edges
    edges = np.zeros((N.count_segs(),2)).astype(int)

    # we are traversing edges in a breadth first manner
    for e in gt.dfs_iterator(N.graph,N.graph.vertex(0)):
        # if the start of the current edge is a start point
        if N.graph.vp['ids'][e.source()] in seg_starts:
            edges[i,0] = N.graph.vp['ids'][e.source()]

        # if the end of the current edge is the end of a segment
        # total degree of the target node in edge
        d = N.graph.vp['degree_total'][e.target()]
        if ((d == 1) | (d > 2)):
            edges[i,1] = N.graph.vp['ids'][e.target()]
            i += 1

    # create subset data frame
    # add root to edges
    edges = np.vstack(([-1,1],edges))
    node_id = edges[:,1]
    parent_id = edges[:,0]

    # subset df
    df = N.node_table[N.node_table.node_id.isin(node_id)].copy()

    # clear types and distance
    df['type'] = df.func.where(df.type != np.int32(0), np.int32(0), df.type)
    df['distance'] = df.func.where(df.distance != np.int32(0), np.int32(0), df.distance)

    # new data frame!
    df2 = vx.from_arrays(node_id = df['node_id'].values,
                        type = df['type'].values,
                        x  = df['x'].values,
                        y = df['y'].values,
                        z = df['z'].values,
                        radius = df['radius'].values,
                        parent_id = parent_id,
                        distance = df['distance'].values)

    # create new graph and add types/distances
    g = graph_from_vaex(df2)

    # get the original path lengths for the new edges?
    dists = np.zeros(len(df))
    nodes = df2['node_id'].values
    for i in g.iter_edges():
        source_id = g.vp['ids'].a[i[0]]
        parent_id = g.vp['ids'].a[i[1]]

        # node in original graph
        s_ind = N.graph.vertex(np.where(N.graph.vp['ids'].a == source_id)[0])
        p_ind = N.graph.vertex(np.where(N.graph.vp['ids'].a == parent_id)[0])
        
        # add distance to dists
        dists[np.where(nodes == parent_id)] = gt.shortest_distance(g = N.graph,source = s_ind,
                                                                    target = p_ind, 
                                                                    weights = N.graph.ep['distance'], 
                                                                    directed = False)

    # add to node table    
    df2['path_length'] = dists

    # remove root 0
    dists = dists[np.where(dists != 0)]

    # add path length property to graph
    eprop_plen = g.new_ep('double')
    eprop_plen.a = dists
    g.ep['path_length'] = eprop_plen

    # create neuron
    N2 = Neuron_Tree(name = N.name, node_table = df2, graph = g)
    # add distances and types
    N2.classify_nodes()
    N2.add_distance()

    return N2