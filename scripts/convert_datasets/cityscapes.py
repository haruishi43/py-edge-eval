#!/usr/bin/env python3

import argparse
from functools import partial
import os

from cityscapesscripts.preparation.json2labelImg import json2labelImg
import numpy as np
from PIL import Image

from pyEdgeEval.common.multi_label.edge_encoding import (
    default_multilabel_encoding,
    rgb_multilabel_encoding,
)
from pyEdgeEval.common.multi_label.dataset_attributes import (
    CITYSCAPES_labelIds,
    CITYSCAPES_label2trainId,
    CITYSCAPES_inst_labelIds,
)
from pyEdgeEval.edge_tools.mask2edge_loop import (
    loop_instance_mask2edge,
    loop_mask2edge,
)
from pyEdgeEval.utils import (
    scandir,
    mkdir_or_exist,
    track_parallel_progress,
    track_progress,
    mask_to_onehot,
    edge_label2trainId,
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
        "--insensitive",
        action="store_true",
        help="default to instance-sensitive, but this argument makes it insensitive",
    )
    parser.add_argument(
        "--only-full-scale",
        action="store_true",
        help="half scales are saved by default; this option disables this",
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
    inst_sensitive: bool = True,
    proc_dir: str = "gtEval",
    label_suffix: str = "_gtFine_labelIds.png",
    inst_suffix: str = "_gtFine_instanceIds.png",
    edge_suffix: str = "_gtProc_edge.png",
    radius: int = 2,
    scale: float = 1.0,
) -> None:
    label_dir = os.path.dirname(label_file)
    label_fn = os.path.basename(label_file)
    proc_dir = label_dir.replace("gtFine", proc_dir)
    proc_fn = label_fn.replace(label_suffix, edge_suffix)

    mkdir_or_exist(proc_dir)
    proc_file = os.path.join(proc_dir, proc_fn)

    # convert function -->
    assert os.path.exists(label_file)
    _, save_format = os.path.splitext(proc_file)
    assert save_format in (".png", ".tif", ".bin")

    label_img = Image.open(label_file)
    # scale if needed
    if scale < 1:
        _img = np.array(label_img)
        h, w = _img.shape
        height, width = int(h * scale + 0.5), int(w * scale + 0.5)
        label_img = label_img.resize((width, height), Image.NEAREST)

    mask = np.array(label_img)

    # NOTE: hard-coded
    ignore_classes = [2, 3]

    # create label mask
    m = mask_to_onehot(mask, labels=CITYSCAPES_labelIds)

    if inst_sensitive:
        inst_file = os.path.join(label_dir, label_fn.replace(label_suffix, inst_suffix))
        assert os.path.exists(inst_file)
        inst_img = Image.open(inst_file)
        if scale < 1:
            inst_img = inst_img.resize((width, height), Image.NEAREST)
        inst_mask = np.array(inst_img)  # int32
        edge_ids = loop_instance_mask2edge(
            mask=m,
            inst_mask=inst_mask,
            inst_labelIds=CITYSCAPES_inst_labelIds,
            ignore_labelIds=ignore_classes,
            radius=radius,
        )
    else:
        edge_ids = loop_mask2edge(
            mask=m,
            ignore_labelIds=ignore_classes,
            radius=radius,
        )

    edge_trainIds = edge_label2trainId(edge=edge_ids, label2trainId=CITYSCAPES_label2trainId)

    # encode and save -->
    if save_format == ".png":
        edges = rgb_multilabel_encoding(edge_trainIds)
        edges_img = Image.fromarray(edges)
        edges_img.save(proc_file)
    elif save_format == ".tif":
        edges = default_multilabel_encoding(edge_trainIds)
        edges = edges.view(np.int32)
        edges_img = Image.fromarray(edges)
        edges_img.save(proc_file)
    elif save_format == ".bin":
        edges = default_multilabel_encoding(edge_trainIds)
        edges.tofile(
            proc_file,
            dtype=np.uint32,
        )
    else:
        raise ValueError()


def convert_split(
    label_files,
    nproc,
    inst_sensitive,
    proc_dir,
    labelIds_suffix,
    instIds_suffix,
    edge_suffix,
    radius,
    scale=1,
):
    """Wrapper function"""
    convert_to_edges = partial(
        convert_label_to_semantic_edges,
        inst_sensitive=inst_sensitive,
        proc_dir=proc_dir,
        label_suffix=labelIds_suffix,
        inst_suffix=instIds_suffix,
        edge_suffix=edge_suffix,
        radius=radius,
        scale=scale,
    )
    if nproc > 1:
        track_parallel_progress(convert_to_edges, label_files, nproc)
    else:
        track_progress(convert_to_edges, label_files)


def test_edges(
    edge_file: str,
    orig_dir: str = "gtProc",  # made with MATLAB script
    test_dir: str = "gtEval",
    orig_suffix: str = "_gtProc_edge.png",
    test_suffix: str = "_gtProc_raw_edge.png",
) -> None:
    edge_dir = os.path.dirname(edge_file)
    edge_fn = os.path.basename(edge_file)
    test_dir = edge_dir.replace(orig_dir, test_dir)
    test_file = os.path.join(test_dir, edge_fn)
    test_file = test_file.replace(orig_suffix, test_suffix)
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

    inst_sensitive = not args.insensitive

    split_names = ["train", "val", "test"]

    # img_suffix = "_leftImg8bit.png"
    # color_suffix = "_gtFine_color.png"
    labelIds_suffix = "_gtFine_labelIds.png"
    instIds_suffix = "_gtFine_instanceIds.png"
    labelTrainIds_suffix = "_gtFine_labelTrainIds.png"
    polygons_suffix = "_gtFine_polygons.json"

    ext = args.ext
    assert ext in ("png", "bin", "tif"), f"extention {ext} is not supported"

    if inst_sensitive:
        print("Instance sensitive")
        raw_edge_name = "raw_isedge"
        thin_edge_name = "thin_isedge"

        train_edge_suffix = f"_gtProc_isedge.{ext}"
        raw_edge_suffix = f"_gtProc_{raw_edge_name}.{ext}"
        thin_edge_suffix = f"_gtProc_{thin_edge_name}.{ext}"
        half_raw_edge_suffix = f"_gtProc_half_{raw_edge_name}.{ext}"
        half_thin_edge_suffix = f"_gtProc_half_{thin_edge_name}.{ext}"
    else:
        print("Instance insensitive")
        raw_edge_name = "raw_edge"
        thin_edge_name = "thin_edge"

        train_edge_suffix = f"_gtProc_edge.{ext}"
        raw_edge_suffix = f"_gtProc_{raw_edge_name}.{ext}"
        thin_edge_suffix = f"_gtProc_{thin_edge_name}.{ext}"
        half_raw_edge_suffix = f"_gtProc_half_{raw_edge_name}.{ext}"
        half_thin_edge_suffix = f"_gtProc_half_{thin_edge_name}.{ext}"

    # hard-coded raw and thin radius
    # NOTE: THIS PART IS IMPORTANT!!!
    # The radius differs for each splits
    # For "raw" evaluation, I think the radius should match
    train_radius = 2
    raw_val_radius = 2
    thin_val_radius = 1

    if not args.test_mode:

        save_half_val = not args.only_full_scale
        if save_half_val:
            print("saving half scale for valuation")

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
        for split in split_names:
            if split == "test":
                continue

            print(f"processsing {split}")
            label_files = []
            gt_split_dir = os.path.join(gt_dir, split)
            for label in scandir(gt_split_dir, labelIds_suffix, recursive=True):
                label_file = os.path.join(gt_split_dir, label)
                label_files.append(label_file)

            convert_full = partial(
                convert_split,
                label_files=label_files,
                nproc=args.nproc,
                inst_sensitive=inst_sensitive,
                proc_dir=proc_dir,
                labelIds_suffix=labelIds_suffix,
                instIds_suffix=instIds_suffix,
            )

            if split == "train":
                # save only full scale
                convert_full(edge_suffix=train_edge_suffix, radius=train_radius)
            elif split == "val":
                # raw
                print("generating raw")
                convert_full(edge_suffix=raw_edge_suffix, radius=raw_val_radius)

                # thin
                print("generating thin")
                convert_full(edge_suffix=thin_edge_suffix, radius=thin_val_radius)

                if save_half_val:
                    convert_half = partial(
                        convert_split,
                        label_files=label_files,
                        nproc=args.nproc,
                        inst_sensitive=inst_sensitive,
                        proc_dir=proc_dir,
                        labelIds_suffix=labelIds_suffix,
                        instIds_suffix=instIds_suffix,
                        scale=0.5,
                    )
                    print("generating half scale raw")
                    convert_half(edge_suffix=half_raw_edge_suffix, radius=raw_val_radius)

                    print("generating half scale thin")
                    convert_half(edge_suffix=half_thin_edge_suffix, radius=thin_val_radius)

        # 3. save split information
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

        assert not inst_sensitive, "can only test for instance insensitive edges"
        print("warning: if validation radius is not 2, there might be a lot of misses")

        orig_name = "gtProc"  # NOTE: hard-coded
        test_name = proc_dir
        orig_edge_suffix = "_gtProc_edge.png"
        val_edge_suffix = "_gtProc_raw_edge.png"

        orig_dir = os.path.join(cityscapes_root, orig_name)

        # train
        split = "train"
        orig_files = []
        orig_split_dir = os.path.join(orig_dir, split)
        for edge in scandir(orig_split_dir, orig_edge_suffix, recursive=True):
            edge_file = os.path.join(orig_split_dir, edge)
            orig_files.append(edge_file)

        test_func = partial(
            test_edges,
            orig_dir=orig_name,
            test_dir=test_name,
            test_suffix=train_edge_suffix,
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
                print(fn, diff)

        print("[train] total missed: ", misses)

        # val
        split = "val"
        orig_files = []
        orig_split_dir = os.path.join(orig_dir, split)
        for edge in scandir(orig_split_dir, orig_edge_suffix, recursive=True):
            edge_file = os.path.join(orig_split_dir, edge)
            orig_files.append(edge_file)

        test_func = partial(
            test_edges,
            orig_dir=orig_name,
            test_dir=test_name,
            test_suffix=val_edge_suffix,
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
                print(fn, diff)

        print("[val] total missed: ", misses)


if __name__ == "__main__":
    main()
