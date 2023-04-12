from .core import *
import graph_tool.all as gt

def classify_segments(N):
    """
    Given a neuron, add segment IDs to graph
    """
    g = N.graph.copy()


    # add total degree property map

    g.vp['degree_total'] = g.degree_property_map("total")

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