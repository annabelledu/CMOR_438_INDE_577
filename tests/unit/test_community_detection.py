import numpy as np
from rice_ml.unsupervised_learning import connected_components

def test_connected_components_edges():
    edges = np.array([[0,1],[1,2],[3,4]])
    comms = connected_components(edges)
    assert comms == [[0,1,2],[3,4]]
