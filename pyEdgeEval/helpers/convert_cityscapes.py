#!/usr/bin/env python3

"""Convert Cityscapes to SBD format

TODO:
- each process uses around 10GB (potentially OOM on constrained systems)
- debug memory usage and unnecessary allocations (probably cv2)
"""

import os.path as osp
from functools import partial

import cv2
import numpy as np
from PIL import Image
from cityscapesscripts.preparation.json2labelImg import json2labelImg

from pyEdgeEval.common.multi_label import (
    default_multilabel_encoding,
    rgb_multilabel_encoding,
)
from pyEdgeEval.edge_tools import (
    loop_instance_mask2edge,
    loop_mask2edge,
)
from pyEdgeEval.utils import (
    edge_label2trainId,
    mask2onehot,
    mkdir_or_exist,
    scandir,
    track_parallel_progress,
    track_progress,
)
from pyEdgeEval.datasets.cityscapes_attributes import (
    CITYSCAPES_labelIds,
    CITYSCAPES_label2trainId,
    CITYSCAPES_inst_labelIds,
)

__all__ = ["convert_cityscapes", "test_against_matlab"]


def convert_json_to_label(
    json_file: str,
    proc_dir: str = "gtEval",
    poly_suffix: str = "_gtFine_polygons.json",
    id_suffix: str = "_gtFine_labelTrainIds.png",
) -> None:
    json_dir = osp.dirname(json_file)
    json_fn = osp.basename(json_file)
    proc_dir = json_dir.replace("gtFine", proc_dir)
    proc_fn = json_fn.replace(poly_suffix, id_suffix)
    mkdir_or_exist(proc_dir)
    proc_file = osp.join(proc_dir, proc_fn)
    json2labelImg(json_file, proc_file, "trainIds")


def resize_np(array, scale=0.5):
    """Resize for numpy array"""
    h, w = array.shape
    oh, ow = int(h * scale + 0.5), int(w * scale + 0.5)
    array = cv2.resize(array, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return array


def convert_label_to_semantic_edges(
    label_file: str,
    inst_sensitive: bool = True,
    proc_dir: str = "gtEval",
    label_suffix: str = "_gtFine_labelIds.png",
    inst_suffix: str = "_gtFine_instanceIds.png",
    edge_suffix: str = "_gtProc_edge.png",
    radius: int = 2,
    thin: bool = False,
    scale: float = 1.0,
    seal: bool = True,
) -> None:
    """Preprocessing Cityscapes GTs

    Args:
        label_file (str)
        inst_sensitive (bool): instance sensitive mode
        proc_dir (str): directory to save the GTs
        label_suffix (str)
        inst_suffix (str)
        edge_suffix (str)
        radius (int): default 2
        thin (bool): default False
        seal (bool):
            Does the same preprocessing as CASENet/SEAL (recommended)
            default True

    Returns:
        None
    """
    if thin:
        assert radius == 1, "ERR: `thin` requires radius=1"
    label_dir = osp.dirname(label_file)
    label_fn = osp.basename(label_file)
    proc_dir = label_dir.replace("gtFine", proc_dir)
    proc_fn = label_fn.replace(label_suffix, edge_suffix)

    mkdir_or_exist(proc_dir)
    proc_file = osp.join(proc_dir, proc_fn)

    # convert function -->
    assert osp.exists(label_file)
    _, save_format = osp.splitext(proc_file)
    assert save_format in (".png", ".tif", ".bin")

    label_img = Image.open(label_file)
    (w, h) = label_img.size

    # scale if needed
    # if "thin", it forces rescaling in the beginning
    if scale < 1 and (not seal or thin):
        height, width = int(h * scale + 0.5), int(w * scale + 0.5)
        label_img = label_img.resize((width, height), Image.Resampling.NEAREST)

    mask = np.array(label_img)

    # NOTE: hard-coded
    ignore_indices = [2, 3]

    # create label mask
    m = mask2onehot(mask, labels=CITYSCAPES_labelIds)

    if inst_sensitive:
        inst_file = osp.join(
            label_dir, label_fn.replace(label_suffix, inst_suffix)
        )
        assert osp.exists(inst_file)
        inst_img = Image.open(inst_file)
        # scale if needed
        # if "thin", it forces rescaling in the beginning
        if scale < 1 and (not seal or thin):
            inst_img = inst_img.resize(
                (width, height), Image.Resampling.NEAREST
            )
        inst_mask = np.array(inst_img)  # int32
        edge_ids = loop_instance_mask2edge(
            mask=m,
            inst_mask=inst_mask,
            inst_labelIds=CITYSCAPES_inst_labelIds,
            ignore_indices=ignore_indices,
            radius=radius,
            thin=thin,
        )
    else:
        edge_ids = loop_mask2edge(
            mask=m,
            ignore_indices=ignore_indices,
            radius=radius,
            thin=thin,
        )

    edge_trainIds = edge_label2trainId(
        edge=edge_ids, label2trainId=CITYSCAPES_label2trainId
    )

    # resize at the end (used in CASENet/SEAL), but don't resize when using "thin"
    if scale < 1 and seal and not thin:
        height, width = int(h * scale + 0.5), int(w * scale + 0.5)
        num_classes = len(CITYSCAPES_label2trainId)
        new_edge_trainIds = np.zeros(
            (num_classes, height, width), dtype=np.uint8
        )
        for cat in range(num_classes):
            new_edge_trainIds[cat, :, :] = resize_np(
                edge_trainIds[cat, :, :], scale=scale
            )
        edge_trainIds = new_edge_trainIds

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
        edges.tofile(proc_file, dtype=np.uint32)
    else:
        raise ValueError()


def convert_routine_wrapper(label_files, nproc=1, **kwargs):
    """Wrapper function

    Args:
        label_files (List[str]): List of files to convert
        nproc (int): number of processes to spawn

    Returns:
        None
    """
    convert_to_edges = partial(convert_label_to_semantic_edges, **kwargs)
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
    edge_dir = osp.dirname(edge_file)
    edge_fn = osp.basename(edge_file)
    test_dir = edge_dir.replace(orig_dir, test_dir)
    test_file = osp.join(test_dir, edge_fn)
    test_file = test_file.replace(orig_suffix, test_suffix)
    _, ext = osp.splitext(edge_file)
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


def convert_cityscapes(
    proc_dirname: str,
    cityscapes_root: str = "data/cityscapes",
    inst_sensitive: bool = True,
    ext: str = "png",
    save_half_val: bool = True,
    train_radius: int = 2,
    raw_val_radius: int = 2,
    thin_val_radius: int = 1,
    nproc: int = 4,
):
    assert proc_dirname, f"proc_dir name is invalid: {proc_dirname}"
    print(f"saving to {cityscapes_root}/{proc_dirname}")
    mkdir_or_exist(osp.join(cityscapes_root, proc_dirname))

    gt_dir = osp.join(cityscapes_root, "gtFine")
    assert osp.exists(gt_dir), f"cannot find {gt_dir}"

    split_names = ["train", "val", "test"]

    # img_suffix = "_leftImg8bit.png"
    # color_suffix = "_gtFine_color.png"
    labelIds_suffix = "_gtFine_labelIds.png"
    instIds_suffix = "_gtFine_instanceIds.png"
    labelTrainIds_suffix = "_gtFine_labelTrainIds.png"
    polygons_suffix = "_gtFine_polygons.json"

    assert ext in ("png", "bin", "tif"), f"extention {ext} is not supported"

    if inst_sensitive:
        print(">>> Instance Sensitive")
        raw_edge_name = "raw_isedge"
        thin_edge_name = "thin_isedge"

        train_edge_suffix = f"_gtProc_isedge.{ext}"
        raw_edge_suffix = f"_gtProc_{raw_edge_name}.{ext}"
        thin_edge_suffix = f"_gtProc_{thin_edge_name}.{ext}"
        half_raw_edge_suffix = f"_gtProc_half_{raw_edge_name}.{ext}"
        half_thin_edge_suffix = f"_gtProc_half_{thin_edge_name}.{ext}"
    else:
        print(">> non-Instance Sensitive")
        raw_edge_name = "raw_edge"
        thin_edge_name = "thin_edge"

        train_edge_suffix = f"_gtProc_edge.{ext}"
        raw_edge_suffix = f"_gtProc_{raw_edge_name}.{ext}"
        thin_edge_suffix = f"_gtProc_{thin_edge_name}.{ext}"
        half_raw_edge_suffix = f"_gtProc_half_{raw_edge_name}.{ext}"
        half_thin_edge_suffix = f"_gtProc_half_{thin_edge_name}.{ext}"

    if save_half_val:
        print(">>> saving half scale for valuation")

    # 1. convert labelIds to labelTrainIds
    poly_files = []
    for split in split_names:
        if split == "test":
            continue
        gt_split_dir = osp.join(gt_dir, split)
        for poly in scandir(gt_split_dir, polygons_suffix, recursive=True):
            poly_file = osp.join(gt_split_dir, poly)
            poly_files.append(poly_file)

    convert_to_trainIds = partial(
        convert_json_to_label,
        proc_dir=proc_dirname,
        poly_suffix=polygons_suffix,
        id_suffix=labelTrainIds_suffix,
    )

    if nproc > 1:
        track_parallel_progress(convert_to_trainIds, poly_files, nproc)
    else:
        track_progress(convert_to_trainIds, poly_files)

    # 2. convert labelIds to edge maps
    for split in split_names:
        if split == "test":
            continue

        print(f">>> processsing {split}")
        label_files = []
        gt_split_dir = osp.join(gt_dir, split)
        for label in scandir(gt_split_dir, labelIds_suffix, recursive=True):
            label_file = osp.join(gt_split_dir, label)
            label_files.append(label_file)

        convert_full = partial(
            convert_routine_wrapper,
            label_files=label_files,
            nproc=nproc,
            inst_sensitive=inst_sensitive,
            proc_dir=proc_dirname,
            label_suffix=labelIds_suffix,
            inst_suffix=instIds_suffix,
        )

        if split == "train":
            # save only full scale
            convert_full(edge_suffix=train_edge_suffix, radius=train_radius)
        elif split == "val":
            # raw
            print(f">>> generating raw {split}")
            convert_full(edge_suffix=raw_edge_suffix, radius=raw_val_radius)

            # thin
            print(f">>> generating thin {split}")
            convert_full(
                edge_suffix=thin_edge_suffix, radius=thin_val_radius, thin=True
            )

            if save_half_val:
                convert_half = partial(
                    convert_routine_wrapper,
                    label_files=label_files,
                    nproc=nproc,
                    inst_sensitive=inst_sensitive,
                    proc_dir=proc_dirname,
                    label_suffix=labelIds_suffix,
                    inst_suffix=instIds_suffix,
                    scale=0.5,
                )
                print(f">>> generating half scale raw {split}")
                convert_half(
                    edge_suffix=half_raw_edge_suffix,
                    radius=raw_val_radius,
                    seal=True,
                )

                print(f">>> generating half scale thin {split}")
                convert_half(
                    edge_suffix=half_thin_edge_suffix,
                    radius=thin_val_radius,
                    thin=True,
                )

    # 3. save split information
    split_save_root = osp.join(cityscapes_root, "splits")
    mkdir_or_exist(split_save_root)
    print(f">>> saving split.txt to {split_save_root}")

    for split in split_names:
        filenames = []
        for poly in scandir(
            osp.join(gt_dir, split), "_polygons.json", recursive=True
        ):
            filenames.append(poly.replace("_gtFine_polygons.json", ""))
        with open(osp.join(split_save_root, f"{split}.txt"), "w") as f:
            f.writelines(f + "\n" for f in filenames)


def test_against_matlab(
    proc_dirname: str,
    matlab_proc_dirname: str = "gtProc",
    cityscapes_root: str = "data/cityscapes",
    inst_sensitive: bool = True,
    ext: str = "png",
    nproc: int = 4,
):
    assert not inst_sensitive, "can only test for instance insensitive edges"
    print(
        ">>> warning: if validation radius is not 2, there might be a lot of misses"
    )

    test_name = proc_dirname
    orig_edge_suffix = "_gtProc_edge.png"
    val_edge_suffix = "_gtProc_raw_edge.png"
    train_edge_suffix = f"_gtProc_edge.{ext}"  # NOTE: nonIS

    orig_dir = osp.join(cityscapes_root, matlab_proc_dirname)

    # train
    split = "train"
    orig_files = []
    orig_split_dir = osp.join(orig_dir, split)
    for edge in scandir(orig_split_dir, orig_edge_suffix, recursive=True):
        edge_file = osp.join(orig_split_dir, edge)
        orig_files.append(edge_file)

    test_func = partial(
        test_edges,
        orig_dir=matlab_proc_dirname,
        test_dir=test_name,
        test_suffix=train_edge_suffix,
    )
    if nproc > 1:
        results = track_parallel_progress(test_func, orig_files, nproc)
    else:
        results = track_progress(test_func, orig_files)

    acceptable_miss = 3
    misses = 0
    for fn, diff, equal in results:
        if not equal:
            misses += 1
        if diff > acceptable_miss:
            print(fn, diff)

    print(">>> [train] total missed: ", misses)

    # val
    split = "val"
    orig_files = []
    orig_split_dir = osp.join(orig_dir, split)
    for edge in scandir(orig_split_dir, orig_edge_suffix, recursive=True):
        edge_file = osp.join(orig_split_dir, edge)
        orig_files.append(edge_file)

    test_func = partial(
        test_edges,
        orig_dir=matlab_proc_dirname,
        test_dir=test_name,
        test_suffix=val_edge_suffix,
    )
    if nproc > 1:
        results = track_parallel_progress(test_func, orig_files, nproc)
    else:
        results = track_progress(test_func, orig_files)

    acceptable_miss = 3
    misses = 0
    for fn, diff, equal in results:
        if not equal:
            misses += 1
        if diff > acceptable_miss:
            print(fn, diff)

    print(">>> [val] total missed: ", misses)
