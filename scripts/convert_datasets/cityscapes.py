#!/usr/bin/env python3

import argparse
from functools import partial
import os

from cityscapesscripts.preparation.json2labelImg import json2labelImg
import numpy as np
from PIL import Image

from pyEdgeEval.datasets.cityscapes.convert_dataset import label2edge
from pyEdgeEval.utils import (
    scandir,
    mkdir_or_exist,
    track_parallel_progress,
    track_progress,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes for Segmentation and Edge Detection"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/cityscapes",
        help="cityscapes directory",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="gtEval",
        help="where to save the processed gts (`gtEval`)",
    )
    parser.add_argument(
        "--ext",
        default="png",
        type=str,
        choices=("png", "bin", "tif"),
        help="format choices=(png, bin, tif)",
    )
    parser.add_argument(
        "--nproc",
        default=4,
        type=int,
        help="number of processes",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def convert_json_to_label(
    json_file: str,
    proc_dir: str = "gtEval",
    poly_suffix: str = "_gtFine_polygons.json",
    id_suffix: str = "_gtFine_labelTrainIds.png",
) -> None:
    json_dir = os.path.dirname(json_file)
    json_fn = os.path.basename(json_file)
    proc_dir = json_dir.replace("gtFine", proc_dir)
    proc_fn = json_fn.replace(poly_suffix, id_suffix)
    mkdir_or_exist(proc_dir)
    proc_file = os.path.join(proc_dir, proc_fn)
    json2labelImg(json_file, proc_file, "trainIds")


def convert_label_to_semantic_edges(
    label_file: str,
    proc_dir: str = "gtEval",
    label_suffix: str = "_gtFine_labelIds.png",
    edge_suffix: str = "_gtProc_edge.png",
    radius: int = 2,
) -> None:
    label_dir = os.path.dirname(label_file)
    label_fn = os.path.basename(label_file)
    proc_dir = label_dir.replace("gtFine", proc_dir)
    proc_fn = label_fn.replace(label_suffix, edge_suffix)
    mkdir_or_exist(proc_dir)
    proc_file = os.path.join(proc_dir, proc_fn)
    label2edge(
        label_path=label_file,
        save_path=proc_file,
        radius=radius,
    )


def test_edges(
    edge_file: str,
    orig_dir: str = "gtProc",  # made with MATLAB script
    test_dir: str = "gtEval",
) -> None:
    edge_dir = os.path.dirname(edge_file)
    edge_fn = os.path.basename(edge_file)
    test_dir = edge_dir.replace(orig_dir, test_dir)
    test_file = os.path.join(test_dir, edge_fn)
    _, ext = os.path.splitext(edge_file)
    if ext == ".png":
        edge = Image.open(edge_file)
        test = Image.open(test_file)
        edge = np.array(edge, dtype=np.uint8)
        test = np.array(test, dtype=np.uint8)
        diff = np.count_nonzero(edge != test)
        is_equal = np.array_equal(edge, test)
        # if diff > 0:
        #     print(f"failed for {edge_file} and {test_file}")
        # if not is_equal:
        #     print(f"failed for {edge_file} and {test_file}")
    elif ext == ".tif":
        edge = Image.open(edge_file)
        test = Image.open(test_file)
        edge = np.array(edge).astype(np.uint32)
        test = np.array(test).astype(np.uint32)
        diff = np.count_nonzero(edge != test)
        is_equal = np.array_equal(edge, test)
        # if diff > 0:
        #     print(f"failed for {edge_file} and {test_file}")
        # if not is_equal:
        #     print(f"failed for {edge_file} and {test_file}")
    else:
        raise ValueError()

    return (edge_file, diff, is_equal)


def main():
    args = parse_args()
    cityscapes_root = args.root
    proc_dir = args.out_dir
    assert proc_dir, f"out_dir is invalid: {proc_dir}"
    print(f"saving to {cityscapes_root}/{proc_dir}")
    mkdir_or_exist(os.path.join(cityscapes_root, proc_dir))

    gt_dir = os.path.join(cityscapes_root, "gtFine")
    assert os.path.exists(gt_dir), f"cannot find {gt_dir}"

    split_names = ["train", "val", "test"]

    # img_suffix = "_leftImg8bit.png"
    # color_suffix = "_gtFine_color.png"
    labelIds_suffix = "_gtFine_labelIds.png"
    # instIds_suffix = "_gtFine_instanceIds.png"
    labelTrainIds_suffix = "_gtFine_labelTrainIds.png"
    polygons_suffix = "_gtFine_polygons.json"

    if args.ext == "png":
        edge_suffix = "_gtProc_edge.png"
    elif args.ext == "bin":
        edge_suffix = "_gtProc_edge.bin"
    elif args.ext == "tif":
        edge_suffix = "_gtProc_edge.tif"
    else:
        raise ValueError()

    if not args.test_mode:

        # 1. convert labelIds to labelTrainIds
        poly_files = []
        for split in split_names:
            if split == "test":
                continue
            gt_split_dir = os.path.join(gt_dir, split)
            for poly in scandir(gt_split_dir, polygons_suffix, recursive=True):
                poly_file = os.path.join(gt_split_dir, poly)
                poly_files.append(poly_file)

        convert_to_trainIds = partial(
            convert_json_to_label,
            proc_dir=proc_dir,
            poly_suffix=polygons_suffix,
            id_suffix=labelTrainIds_suffix,
        )

        if args.nproc > 1:
            track_parallel_progress(convert_to_trainIds, poly_files, args.nproc)
        else:
            track_progress(convert_to_trainIds, poly_files)

        # 2. convert labelIds to edge maps
        label_files = []
        for split in split_names:
            if split == "test":
                continue
            gt_split_dir = os.path.join(gt_dir, split)
            for label in scandir(gt_split_dir, labelIds_suffix, recursive=True):
                label_file = os.path.join(gt_split_dir, label)
                label_files.append(label_file)

        convert_to_edges = partial(
            convert_label_to_semantic_edges,
            proc_dir=proc_dir,
            label_suffix=labelIds_suffix,
            edge_suffix=edge_suffix,
            radius=2,  # NOTE: hard-coded
        )

        if args.nproc > 1:
            track_parallel_progress(convert_to_edges, label_files, args.nproc)
        else:
            track_progress(convert_to_edges, label_files)

        # save split information
        split_save_root = os.path.join(cityscapes_root, "splits")
        mkdir_or_exist(split_save_root)

        for split in split_names:
            filenames = []
            for poly in scandir(
                os.path.join(gt_dir, split), "_polygons.json", recursive=True
            ):
                filenames.append(poly.replace("_gtFine_polygons.json", ""))
            with open(os.path.join(split_save_root, f"{split}.txt"), "w") as f:
                f.writelines(f + "\n" for f in filenames)

    else:
        print("testing!")

        orig_name = "gtProc"  # NOTE: hard-coded
        test_name = proc_dir

        orig_dir = os.path.join(cityscapes_root, orig_name)

        orig_files = []
        for split in split_names:
            if split == "test":
                continue
            orig_split_dir = os.path.join(orig_dir, split)
            for edge in scandir(orig_split_dir, edge_suffix, recursive=True):
                edge_file = os.path.join(orig_split_dir, edge)
                orig_files.append(edge_file)

        test_func = partial(
            test_edges,
            orig_dir=orig_name,
            test_dir=test_name,
        )

        if args.nproc > 1:
            results = track_parallel_progress(test_func, orig_files, args.nproc)
        else:
            results = track_progress(test_func, orig_files)

        acceptable_miss = 3
        misses = 0
        for fn, diff, equal in results:
            if not equal:
                misses += 1
            if diff > acceptable_miss:
                print(fn)

        print("total missed: ", misses)


if __name__ == "__main__":
    main()
