"""
Community Detection (NumPy-only).

This module provides a simple, deterministic "community detection" baseline:
connected components on an (un)directed graph.

Why this approach?
- It is easy to understand and test.
- It is a solid building block for more advanced community methods.

Supported inputs
----------------
1) Adjacency matrix A of shape (n, n)
   - nonzero entries indicate edges (optionally weighted).
2) Edge list edges of shape (m, 2)
   - each row [u, v] indicates an edge.

Functions
---------
- connected_components(...)
- community_detection(...)  (alias)

Outputs
-------
- communities: List[List[int]] sorted communities
- (optional) labels: ndarray shape (n,) with component ids 0..k-1

Notes
-----
- Self-loops are ignored.
- For undirected graphs, edges are treated symmetrically.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[int]]]


def _as_int_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.integer):
        # allow floats like 0.0/1.0 for adjacency matrices, but cast safely
        if np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(int)
        else:
            raise TypeError(f"{name} must be numeric/integer.")
    return arr


def _validate_adjacency(A: np.ndarray) -> np.ndarray:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be square with shape (n, n).")
    if A.shape[0] == 0:
        raise ValueError("Adjacency matrix A must be non-empty.")
    return A


def _build_adj_from_edges(edges: np.ndarray, n_nodes: Optional[int], directed: bool) -> np.ndarray:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2).")
    if edges.shape[0] == 0:
        raise ValueError("edges must be non-empty.")

    if n_nodes is None:
        n_nodes = int(np.max(edges)) + 1
    if n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer.")
    if np.min(edges) < 0:
        raise ValueError("edges must contain non-negative node indices.")

    A = np.zeros((n_nodes, n_nodes), dtype=int)
    # fill
    for u, v in edges:
        if u == v:
            continue
        A[u, v] = 1
        if not directed:
            A[v, u] = 1
    return A


def connected_components(
    graph: ArrayLike,
    *,
    n_nodes: Optional[int] = None,
    directed: bool = False,
    return_labels: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], np.ndarray]]:
    """
    Compute connected components ("communities") of a graph.

    Parameters
    ----------
    graph : array_like
        Either an adjacency matrix A (n,n) or an edge list (m,2).
    n_nodes : int or None, optional
        Only used if graph is an edge list and node count cannot be inferred.
    directed : bool, default=False
        If True and graph is edge list, treat edges as directed.
        For adjacency matrices, directed=True uses edges as given.
        If directed=False, adjacency is symmetrized (A|A^T).
    return_labels : bool, default=False
        If True, also return an array labels[i] = component_id.

    Returns
    -------
    communities : list of list of int
        Each community is a sorted list of node indices.
    labels : ndarray of shape (n,), optional
        Component id per node (0..k-1) if return_labels=True.
    """
    arr = np.asarray(graph)

    # Determine input form
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        A = _validate_adjacency(_as_int_array(arr, "graph"))
        if not directed:
            A = ((A != 0) | (A.T != 0)).astype(int)
        else:
            A = (A != 0).astype(int)
        n = A.shape[0]
    else:
        edges = _as_int_array(arr, "graph")
        A = _build_adj_from_edges(edges, n_nodes=n_nodes, directed=directed)
        n = A.shape[0]

    # BFS/DFS over adjacency matrix rows
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    communities: List[List[int]] = []
    comp_id = 0

    for start in range(n):
        if visited[start]:
            continue

        # start a new component
        queue = [start]
        visited[start] = True
        labels[start] = comp_id
        members = []

        qpos = 0
        while qpos < len(queue):
            u = queue[qpos]
            qpos += 1
            members.append(int(u))

            nbrs = np.where(A[u] != 0)[0]
            for v in nbrs.tolist():
                if not visited[v]:
                    visited[v] = True
                    labels[v] = comp_id
                    queue.append(int(v))

        members.sort()
        communities.append(members)
        comp_id += 1

    # Sort communities by (size desc, then lexicographic) for determinism
    communities.sort(key=lambda c: (-len(c), c))

    if return_labels:
        # After sorting communities, remap labels to match sorted order
        remap = {}
        for new_id, comm in enumerate(communities):
            for node in comm:
                remap[node] = new_id
        new_labels = np.array([remap[i] for i in range(n)], dtype=int)
        return communities, new_labels

    return communities


def community_detection(
    graph: ArrayLike,
    *,
    n_nodes: Optional[int] = None,
    directed: bool = False,
    return_labels: bool = False,
):
    """Alias for connected_components."""
    return connected_components(graph, n_nodes=n_nodes, directed=directed, return_labels=return_labels)
