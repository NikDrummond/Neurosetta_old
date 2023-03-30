# testing core object properties based on test data

from Neurosetta.core import read_swc, classify_nodes
import pytest
import vaex as vx
import numpy as np

@pytest.fixture
def data():
    return read_swc('tests/core/test_data.swc')

def test_swc_import(data):
    """"""
    # check type
    assert isinstance(data,vx.DataFrame)
    # check shape
    assert data.shape == (50,7)


def test_node_counts(data):
    """"""
    test_data = classify_nodes(data, overwrite = True)
    # test number of branches
    assert len(data[data.type == 5]) == 14
    # test number of ends
    assert len(data[data.type == 6]) == 19
    # test number of roots
    assert len(data[data.type == 1]) == 1

    # test typing
    answer = np.array([1,5,6,0,5,6,5,
                       0,0,0,5,6,6,0,
                       0,5,0,6,5,6,6,
                       5,0,0,5,5,6,6,
                       0,6,6,0,0,0,5,
                       6,6,6,5,0,5,6,
                       6,0,5,5,0,6,6,
                       6])
    assert np.array_equal(data['type'].values,answer)