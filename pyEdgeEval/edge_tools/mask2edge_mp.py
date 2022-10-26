#!/usr/bin/env python3

"""
NOTE: not much speed ups probably because allocating memory is slower than
the gain from multiprocessing

Might be beneficial when each worker has more things to do...

TODO:
- add `thin`
"""

from multiprocessing import Pool, RawArray

import numpy as np

from pyEdgeEval.utils import mask2bdry

__all__ = [
    "mp_mask2edge",
    "mp_instance_mask2edge",
]

# HACK: A global dictionary storing the variables passed from the initializer.
var_dict = {}


def _init_worker(data_dict):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.

    for k, v in data_dict.items():
        var_dict[k] = v


def _worker_func(label):
    mask_shape = var_dict["mask_shape"]
    mask_dtype = var_dict["mask_dtype"]
    edge_shape = var_dict["edge_shape"]
    edge_dtype = var_dict["edge_dtype"]
    ignore_mask_shape = var_dict["ignore_mask_shape"]
    ignore_mask_dtype = var_dict["ignore_mask_dtype"]
    ignore_indices = var_dict["ignore_indices"]
    mask2bdry_kwargs = var_dict["mask2bdry_kwargs"]

    mask = np.frombuffer(var_dict["mask"], dtype=mask_dtype).reshape(mask_shape)
    ignore_mask = np.frombuffer(
        var_dict["ignore_mask"], dtype=ignore_mask_dtype
    ).reshape(ignore_mask_shape)
    edge = np.frombuffer(var_dict["edge"], dtype=edge_dtype).reshape(edge_shape)

    mask = mask[label]

    if label in ignore_indices:
        return None

    # if there are no class labels in the mask
    if not np.count_nonzero(mask):
        return None

    edge[label] = mask2bdry(
        mask=mask,
        ignore_mask=ignore_mask,
        **mask2bdry_kwargs,
    )
    return None


def mp_mask2edge(
    mask,
    ignore_indices,
    nproc,
    radius,
    thin=False,
    use_cv2=True,
    quality=0,
):
    """mask2edge multiprocessing"""

    if thin:
        raise NotImplementedError(
            "thin=True has not been implemented for this function."
        )

    global var_dict

    assert mask.ndim == 3
    num_labels, h, w = mask.shape

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_indices:
        ignore_mask += mask[i]

    mask2bdry_kwargs = dict(
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    edge = np.zeros_like(mask)

    # create shared arrays
    mp_mask = RawArray(np.ctypeslib.as_ctypes_type(mask.dtype), mask.size)
    mp_edge = RawArray(np.ctypeslib.as_ctypes_type(edge.dtype), edge.size)
    mp_ignore_mask = RawArray(
        np.ctypeslib.as_ctypes_type(ignore_mask.dtype), ignore_mask.size
    )

    # wrap arrays as an numpy array so we can easily manipulate data
    np_mask = np.frombuffer(mp_mask, dtype=mask.dtype).reshape(mask.shape)
    np_edge = np.frombuffer(mp_edge, dtype=edge.dtype).reshape(edge.shape)
    np_ignore_mask = np.frombuffer(
        mp_ignore_mask, dtype=ignore_mask.dtype
    ).reshape(ignore_mask.shape)

    # copy data to wrapped shared arrays
    np.copyto(np_mask, mask)
    np.copyto(np_edge, edge)
    np.copyto(np_ignore_mask, ignore_mask)

    args = dict(
        mask=mp_mask,
        mask_dtype=mask.dtype,
        mask_shape=mask.shape,
        edge=mp_edge,
        edge_dtype=edge.dtype,
        edge_shape=edge.shape,
        ignore_mask=mp_ignore_mask,
        ignore_mask_dtype=ignore_mask.dtype,
        ignore_mask_shape=ignore_mask.shape,
        ignore_indices=ignore_indices,
        mask2bdry_kwargs=mask2bdry_kwargs,
    )

    with Pool(
        processes=nproc, initializer=_init_worker, initargs=[args]
    ) as pool:
        # order doesn't really matter since we specify which index to compute
        pool.map(_worker_func, range(num_labels))

    edge = np.frombuffer(mp_edge, dtype=edge.dtype).reshape(edge.shape)

    # HACK: empty global dictionary
    var_dict = {}

    return edge


def _instance_worker_func(label):
    mask_shape = var_dict["mask_shape"]
    mask_dtype = var_dict["mask_dtype"]
    inst_mask_shape = var_dict["inst_mask_shape"]
    inst_mask_dtype = var_dict["inst_mask_dtype"]
    edge_shape = var_dict["edge_shape"]
    edge_dtype = var_dict["edge_dtype"]
    ignore_mask_shape = var_dict["ignore_mask_shape"]
    ignore_mask_dtype = var_dict["ignore_mask_dtype"]
    ignore_indices = var_dict["ignore_indices"]
    label_inst = var_dict["label_inst"]
    mask2bdry_kwargs = var_dict["mask2bdry_kwargs"]

    mask = np.frombuffer(var_dict["mask"], dtype=mask_dtype).reshape(mask_shape)
    inst_mask = np.frombuffer(
        var_dict["inst_mask"], dtype=inst_mask_dtype
    ).reshape(inst_mask_shape)
    ignore_mask = np.frombuffer(
        var_dict["ignore_mask"], dtype=ignore_mask_dtype
    ).reshape(ignore_mask_shape)
    edge = np.frombuffer(var_dict["edge"], dtype=edge_dtype).reshape(edge_shape)

    mask = mask[label]

    if label in ignore_indices:
        return None

    # if there are no class labels in the mask
    if not np.count_nonzero(mask):
        return None

    dist = mask2bdry(
        mask=mask,
        ignore_mask=ignore_mask,
        **mask2bdry_kwargs,
    )

    # per instance boundaries
    if label in label_inst.keys():
        instances = label_inst[label]  # list
        for instance in instances:
            iid = int(str(label) + str(instance).zfill(3))
            _mask = inst_mask == iid
            dist = dist | mask2bdry(
                mask=_mask,
                ignore_mask=ignore_mask,
                **mask2bdry_kwargs,
            )

    # set edge
    edge[label] = dist
    return None


def mp_instance_mask2edge(
    mask,
    inst_mask,
    inst_labelIds,
    ignore_indices,
    nproc,
    radius,
    thin=False,
    use_cv2=True,
    quality=0,
):
    """mask2edge multiprocessing"""

    if thin:
        raise NotImplementedError(
            "thin=True has not been implemented for this function."
        )

    global var_dict

    assert mask.ndim == 3
    num_labels, h, w = mask.shape

    # make ignore mask
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    for i in ignore_indices:
        ignore_mask += mask[i]

    # make sure that instance labels are sorted
    inst_labels = sorted(inst_labelIds)

    # create a lookup dictionary {label: [instances]}
    label_inst = {}
    _cand_insts = np.unique(inst_mask)
    for inst_label in _cand_insts:
        if inst_label < inst_labels[0] * 1000:  # 24000
            continue
        _label = int(str(inst_label)[:2])
        _inst = int(str(inst_label)[2:])
        if _label not in label_inst.keys():
            label_inst[_label] = [_inst]
        else:
            label_inst[_label].append(_inst)

    mask2bdry_kwargs = dict(
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )

    edge = np.zeros_like(mask)

    # create shared arrays
    mp_mask = RawArray(np.ctypeslib.as_ctypes_type(mask.dtype), mask.size)
    mp_inst_mask = RawArray(
        np.ctypeslib.as_ctypes_type(inst_mask.dtype), inst_mask.size
    )
    mp_edge = RawArray(np.ctypeslib.as_ctypes_type(edge.dtype), edge.size)
    mp_ignore_mask = RawArray(
        np.ctypeslib.as_ctypes_type(ignore_mask.dtype), ignore_mask.size
    )

    # wrap arrays as an numpy array so we can easily manipulate data
    np_mask = np.frombuffer(mp_mask, dtype=mask.dtype).reshape(mask.shape)
    np_inst_mask = np.frombuffer(mp_inst_mask, dtype=inst_mask.dtype).reshape(
        inst_mask.shape
    )
    np_edge = np.frombuffer(mp_edge, dtype=edge.dtype).reshape(edge.shape)
    np_ignore_mask = np.frombuffer(
        mp_ignore_mask, dtype=ignore_mask.dtype
    ).reshape(ignore_mask.shape)

    # copy data to wrapped shared arrays
    np.copyto(np_mask, mask)
    np.copyto(np_inst_mask, inst_mask)
    np.copyto(np_edge, edge)
    np.copyto(np_ignore_mask, ignore_mask)

    args = dict(
        mask=mp_mask,
        mask_dtype=mask.dtype,
        mask_shape=mask.shape,
        inst_mask=mp_inst_mask,
        inst_mask_dtype=inst_mask.dtype,
        inst_mask_shape=inst_mask.shape,
        edge=mp_edge,
        edge_dtype=edge.dtype,
        edge_shape=edge.shape,
        ignore_mask=mp_ignore_mask,
        ignore_mask_dtype=ignore_mask.dtype,
        ignore_mask_shape=ignore_mask.shape,
        ignore_indices=ignore_indices,
        label_inst=label_inst,
        mask2bdry_kwargs=mask2bdry_kwargs,
    )

    with Pool(
        processes=nproc, initializer=_init_worker, initargs=[args]
    ) as pool:
        # order doesn't really matter since we specify which index to compute
        pool.map(_instance_worker_func, range(num_labels))

    edge = np.frombuffer(mp_edge, dtype=edge.dtype).reshape(edge.shape)

    # HACK: empty global dictionary
    var_dict = {}

    return edge
