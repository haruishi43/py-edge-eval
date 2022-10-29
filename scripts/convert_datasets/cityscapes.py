#!/usr/bin/env python3

import argparse

from pyEdgeEval.helpers.convert_cityscapes import convert_cityscapes, test_against_matlab


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
        "--nonIS",
        action="store_true",
        help="default to IS, but this argument makes it nonIS",
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


def main():
    args = parse_args()

    if not args.test_mode:
        convert_cityscapes(
            proc_dirname=args.out_dir,
            cityscapes_root=args.root,
            inst_sensitive=(not args.nonIS),
            ext=args.ext,
            save_half_val=(not args.only_full_scale),
            train_radius=2,
            raw_val_radius=2,
            thin_val_radius=1,
            nproc=args.nproc,
        )
    else:
        print(">>> testing!")
        test_against_matlab(
            proc_dirname=args.out_dir,
            matlab_proc_dirname="gtProc",
            cityscapes_root=args.root,
            inst_sensitive=(not args.nonIS),
            ext=args.ext,
            nproc=args.nproc,
        )


if __name__ == "__main__":
    main()
