# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from config import get_eval_args
import random
random.seed(1234)
import os
import pickle
from utils.metrics import compute_metrics
import argparse


def computeAverageMetrics(imfeats, recipefeats, k, t, forceorder=False):
    """Computes retrieval metrics for two sets of features

    Parameters
    ----------
    imfeats : np.ndarray [n x d]
        The image features..
    recipefeats : np.ndarray [n x d]
        The recipe features.
    k : int
        Ranking size.
    t : int
        Number of evaluations to run (function returns the average).
    forceorder : bool
        Whether to force a particular order instead of picking random samples

    Returns
    -------
    dict
        Dictionary with metric values for all t runs.

    """

    glob_metrics = {}
    i = 0
    for _ in range(t):

        if forceorder:
            # pick the same samples in the same order for evaluation
            # forceorder is only True when the function is used during training
            sub_ids = np.array(range(i, i + k))
            i += k
        else:
            sub_ids = random.sample(range(0, len(imfeats)), k)
        imfeats_sub = imfeats[sub_ids, :]
        recipefeats_sub = recipefeats[sub_ids, :]

        metrics = compute_metrics(imfeats_sub, recipefeats_sub,
                                  recall_klist=(1, 5, 10))

        for metric_name, metric_value in metrics.items():
            if metric_name not in glob_metrics:
                glob_metrics[metric_name] = []
            glob_metrics[metric_name].append(metric_value)
    return glob_metrics


def eval(args):

    # Load embeddings
    with open(args.embeddings_file, 'rb') as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)

    # sort by name so that we always pick the same samples
    idxs = np.argsort(ids)
    ids = ids[idxs]
    recipefeats = recipefeats[idxs]
    imfeats = imfeats[idxs]

    if args.retrieval_mode == 'image2recipe':
        glob_metrics = computeAverageMetrics(imfeats, recipefeats, args.medr_N, args.ntimes)
    else:
        glob_metrics = computeAverageMetrics(recipefeats, imfeats, args.medr_N, args.ntimes)

    for k, v in glob_metrics.items():
        print (k + ':', np.mean(v))

if __name__ == "__main__":

    args = get_eval_args()
    eval(args)
