#!/usr/bin/env python3

"""Convert SBD dataset

- Convert edges into OTF-compatible edges
- Instance-Sensitive (IS) and Instance-insensitive (nonIS)
- Also saves segmentation (just in case)
- we ignore the pixels around the image boundary (5px); added another label (21)
"""

import argparse
import os.path as osp
from functools import partial

from pyEdgeEval.helpers.convert_sbd import (
    check_path,
    get_samples,
    load_reanno_samples,
    routine,
    reanno_routine,
)
from pyEdgeEval.utils import mkdir_or_exist, track_parallel_progress, track_progress


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SBD for Segmentation and Edge Detection"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/sbd",
        help="SBD directory",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="gtEval",
        help="where to save the processed gts (`gtEval`)",
    )
    parser.add_argument(
        "--nonIS",
        action="store_true",
        help="default to IS, but this argument makes it nonIS",
    )
    parser.add_argument(
        "--nproc",
        default=4,
        type=int,
        help="number of processes",
    )
    parser.add_argument(
        "--reanno",
        action="store_true",
        help="process reannotated GTs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    splits = ('train', 'val')

    data_dir = args.root
    save_dir = osp.join(data_dir, 'gtEval')
    mkdir_or_exist(save_dir)
    nonIS = args.nonIS
    nproc = args.nproc
    reanno = args.reanno
    radius = 2

    raw_edge_suffix = "_raw_edge.png"
    raw_isedge_suffix = "_raw_isedge.png"
    thin_edge_suffix = "_thin_edge.png"
    thin_isedge_suffix = "_thin_isedge.png"

    label_suffix = "_labelIds.png"
    train_suffix = "_trainIds.png"
    inst_suffix = "_instanceIds.png"

    print(">>> root:  \t", data_dir)
    print(">>> save:  \t", save_dir)
    print(">>> nonIS: \t", nonIS)
    print(">>> nproc: \t", nproc)
    print(">>> radius:\t", radius)
    print(">>> reanno:\t", reanno)

    raw_routine = partial(
        routine,
        data_dir=data_dir,
        save_dir=save_dir,
        radius=radius,
        thin=False,
        edge_suffix=raw_edge_suffix,
        isedge_suffix=raw_isedge_suffix,
        label_suffix=label_suffix,
        train_suffix=train_suffix,
        inst_suffix=inst_suffix,
        save_segs=True,
    )
    thin_routine = partial(
        routine,
        data_dir=data_dir,
        save_dir=save_dir,
        radius=radius,
        thin=True,
        edge_suffix=thin_edge_suffix,
        isedge_suffix=thin_isedge_suffix,
        label_suffix=label_suffix,
        train_suffix=train_suffix,
        inst_suffix=inst_suffix,
        save_segs=False,  # already saved by raw
    )

    # process original dataset
    for split in splits:
        sample_names = get_samples(osp.join(data_dir, f"{split}.txt"))
        assert len(sample_names) > 0, f"ERR: should have more than 1 samples, but got {len(sample_names)}"

        if nproc > 1:
            print(">>> ", split, " raw")
            track_parallel_progress(raw_routine, sample_names, nproc)
            print(">>> ", split, " thin")
            track_parallel_progress(thin_routine, sample_names, nproc)
        else:
            print(">>> ", split, " raw")
            track_progress(raw_routine, sample_names)
            print(">>> ", split, " thin")
            track_progress(thin_routine, sample_names)

    # re-annotated GT (SEAL)
    if reanno:
        reanno_raw_dir = check_path(osp.join(data_dir, 'gt_reanno_raw'))
        reanno_thin_dir = check_path(osp.join(data_dir, 'gt_reanno_thin'))

        raw_reanno_edge_suffix = "_reanno_raw_edge.png"
        raw_reanno_isedge_suffix = "_reanno_raw_isedge.png"
        thin_reanno_edge_suffix = "_reanno_thin_edge.png"
        thin_reanno_isedge_suffix = "_reanno_thin_isedge.png"

        raw_reanno_routine = partial(
            reanno_routine,
            data_dir=reanno_raw_dir,
            save_dir=save_dir,
            edge_suffix=raw_reanno_edge_suffix,
            isedge_suffix=raw_reanno_isedge_suffix,
        )
        thin_reanno_routine = partial(
            reanno_routine,
            data_dir=reanno_thin_dir,
            save_dir=save_dir,
            edge_suffix=thin_reanno_edge_suffix,
            isedge_suffix=thin_reanno_isedge_suffix,
        )

        raw_list = load_reanno_samples(osp.join(reanno_raw_dir, 'test.mat'))
        thin_list = load_reanno_samples(osp.join(reanno_thin_dir, 'test.mat'))

        if nproc > 1:
            print(">>> reanno raw")
            track_parallel_progress(raw_reanno_routine, raw_list, nproc)
            print(">>> reanno thin")
            track_parallel_progress(thin_reanno_routine, thin_list, nproc)
        else:
            print(">>> reanno raw")
            track_progress(raw_reanno_routine, raw_list)
            print(">>> reanno thin")
            track_progress(thin_reanno_routine, thin_list)

        # save sample names as txt file
        with open(osp.join(save_dir, 'reanno_val.txt'), 'w') as f:
            f.writelines(s + "\n" for s in raw_list)

    print(">>> done!")
