# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent, down_sampling):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        sampled_feat = np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
    if down_sampling > 1:
        n = sampled_feat.shape[0]
        m = sampled_feat.shape[1]
        sampled_feat = sampled_feat[:(n // down_sampling) * down_sampling, :]
        sampled_feat = sampled_feat.reshape((n // down_sampling, down_sampling * m))

    return sampled_feat


def load_feature(feat_dir, splits, nshard, seed, percent, down_sampling):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent, down_sampling)
            for r in range(nshard) for split in splits.split(',')
        ],
        axis=0,
    )
    np.random.shuffle(feat)
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
        feat_dir,
        splits,
        nshard,
        km_path,
        n_clusters,
        seed,
        percent,
        init,
        max_iter,
        batch_size,
        tol,
        n_init,
        reassignment_ratio,
        max_no_improvement,
        down_sampling,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, splits, nshard, seed, percent, down_sampling)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("splits", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    parser.add_argument("--down_sampling", default=1, type=int)
    args = parser.parse_args()
    if args.max_no_improvement == -1:
        args.max_no_improvement = None
    logging.info(str(args))
    learn_kmeans(**vars(args))
