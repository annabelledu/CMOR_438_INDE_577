import numpy as np
from rice_ml.unsupervised_learning import DBSCAN

def test_dbscan_two_clusters():
    X = np.vstack([
        np.random.randn(20, 2) * 0.1,
        np.random.randn(20, 2) * 0.1 + 5
    ])
    labels = DBSCAN(eps=0.3, min_samples=3).fit_predict(X)
    clusters = set(labels)
    clusters.discard(-1)
    assert len(clusters) == 2
