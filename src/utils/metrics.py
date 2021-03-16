# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Retrieval metrics for aligned data
"""
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_metrics(queries, database, metric='cosine',
                    recall_klist=(1, 5, 10), return_raw=False):
    """Function to compute Median Rank and Recall@k metrics given two sets of
       aligned embeddings.

    Args:
        queries (numpy.ndarray): A NxD dimensional array containing query
                                 embeddings.
        database (numpy.ndarray): A NxD dimensional array containing
                                  database embeddings.
        metric (str): The distance metric to use to compare embeddings.
        recall_klist (list): A list of integers with the k-values to
                             compute recall at.

    Returns:
        metrics (dict): A dictionary with computed values for each metric.
    """
    assert isinstance(queries, np.ndarray), "queries must be of type numpy.ndarray"
    assert isinstance(database, np.ndarray), "database must be of type numpy.ndarray"
    assert queries.shape == database.shape, "queries and database must have the same shape"
    assert len(recall_klist) > 0, "recall_klist cannot be empty"

    # largest k to compute recall
    max_k = int(max(recall_klist))

    assert all(i >= 1 for i in recall_klist), "all values in recall_klist must be at least 1"
    assert max_k <= queries.shape[0], "the highest element in recall_klist must be lower than database.shape[0]"
    if any(isinstance(i, float) for i in recall_klist):
        warnings.warn("All values in recall_klist should be integers. Using int(k) for all values in recall_klist.")

    dists = pairwise_distances(queries, database, metric=metric)

    # find the number of elements in the ranking that have a lower distance
    # than the positive element (whose distance is in the diagonal
    # of the distance matrix) wrt the query. this gives the rank for each
    # query. (+1 for 1-based indexing)
    positions = np.count_nonzero(dists < np.diag(dists)[:, None], axis=-1) + 1

    # get the topk elements for each query (topk elements with lower dist)
    rankings = np.argpartition(dists, range(max_k), axis=-1)[:, :max_k]

    # positive positions for each query (inputs are assumed to be aligned)
    positive_idxs = np.array(range(dists.shape[0]))
    # matrix containing a cumulative sum of topk matches for each query
    # if cum_matches_topk[q][k] = 1, it means that the positive for query q
    # was already found in position <=k. if not, the value at that position
    # will be 0.
    cum_matches_topk = np.cumsum(rankings == positive_idxs[:, None],
                                 axis=-1)

    # pre-compute all possible recall values up to k
    recall_values = np.mean(cum_matches_topk, axis=0)

    metrics = {}
    metrics['medr'] = np.median(positions)

    for index in recall_klist:
        metrics[f'recall_{int(index)}'] = recall_values[int(index)-1]

    if return_raw:
        return metrics, {'medr': positions, 'recall': cum_matches_topk}
    return metrics
