#!/usr/bin/env python3

import argparse
import os.path as osp
import time

from pyEdgeEval.evaluators.bsds import BSDS500Evaluator
from pyEdgeEval.utils import get_root_logger, mkdir_or_exist

__all__ = ["evaluate_bsds500"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BSDS output")
    parser.add_argument(
        "bsds_path",
        type=str,
        help="the root path of the BSDS-500 dataset",
    )
    parser.add_argument(
        "pred_path",
        type=str,
        help="the root path of the predictions",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="the root path of where the results are populated",
    )
    parser.add_argument(
        "--use-val",
        action="store_true",
        help="val or test",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=0.0075,
        help="tolerance distance (default: 0.0075)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="99",
        help="the number of thresholds (could be a list of floats); use 99 for eval",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="option to remove the thinning process (i.e. uses raw predition)",
    )
    parser.add_argument(
        "--apply-nms",
        action="store_true",
        help="applies NMS before evaluation",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="the number of parallel threads",
    )

    return parser.parse_args()


def evaluate(
    bsds_path: str,
    pred_path: str,
    output_path: str,
    use_val: bool,
    max_dist: float,
    thresholds: str,
    apply_thinning: bool,
    apply_nms: bool,
    nproc: int,
    no_split_dir: bool = False,
):
    """Evaluate BSDS500"""

    thresholds = thresholds.strip()
    try:
        n_thresholds = int(thresholds)
        thresholds = n_thresholds
    except ValueError:
        try:
            if thresholds.startswith("[") and thresholds.endswith("]"):
                thresholds = thresholds[1:-1]
                thresholds = [float(t.strip()) for t in thresholds.split(",")]
            else:
                print(
                    "Bad threshold format; should be a python list of floats (`[a, b, c]`)"
                )
                return
        except ValueError:
            print(
                "Bad threshold format; should be a python list of ints (`[a, b, c]`)"
            )
            return

    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_path = osp.join(
            osp.normpath(pred_path), f"edge_results_{timestamp}"
        )

    mkdir_or_exist(output_path)

    split = "val" if use_val else "test"

    # setup logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(output_path, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    logger.info("Running BSDS5000 Evaluation")
    logger.info(f"split:                  \t{split}")
    logger.info(f"thresholds:             \t{thresholds}")
    logger.info(f"max_dist:               \t{max_dist}")
    logger.info(f"thinning + thinned gts: \t{apply_thinning}")
    logger.info(f"nms:                    \t{apply_nms}")
    print("\n\n")

    evaluator = BSDS500Evaluator(
        dataset_root=bsds_path,
        pred_root=pred_path,
        split=split,
    )

    evaluator.set_eval_params(
        scale=1.0,
        apply_thinning=apply_thinning,
        apply_nms=apply_nms,
        max_dist=max_dist,
    )

    evaluator.evaluate(
        thresholds=thresholds,
        nproc=nproc,
        save_dir=output_path,
        no_split_dir=no_split_dir,
    )


def evaluate_bsds500(no_split_dir=False):
    args = parse_args()

    apply_thinning = not args.raw

    evaluate_bsds500(
        bsds_path=args.bsds_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        use_val=args.use_val,
        max_dist=args.max_dist,
        thresholds=args.thresholds,
        apply_thinning=apply_thinning,
        apply_nms=args.apply_nms,
        nproc=args.nproc,
        no_split_dir=no_split_dir,
    )
