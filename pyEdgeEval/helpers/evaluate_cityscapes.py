#!/usr/bin/env python3

"""General evaluation protocols for Cityscapes Semantic Boundary Benchmark

**The 'Raw' Protocol:**
- Use `CityscapesEvaluator` or `HalfcityscapesEvaluator`

**The 'Thin' Protocol:**
- Use `HalfCityscapesEvaluator`

"""

import argparse
import os.path as osp
import time
import warnings

from pyEdgeEval.evaluators import CityscapesEvaluator, HalfCityscapesEvaluator
from pyEdgeEval.utils import get_root_logger, mkdir_or_exist

__all__ = ["evaluate_cityscapes_thin", "evaluate_cityscapes_raw"]


def _common_parser_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "cityscapes_path",
        type=str,
        help="the root path of the cityscapes dataset",
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
        "--categories",
        type=str,
        help="the category number to evaluate; can be multiple values'[1, 14]'",
    )
    parser.add_argument(
        "--pre-seal",
        action="store_true",
        help="prior to SEAL, the evaluations were not as strict",
    )
    parser.add_argument(
        "--nonIS",
        action="store_true",
        help="non instance sensitive evaluation",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=0.0035,
        help="tolerance distance (default: 0.0035)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="99",
        help="the number of thresholds (could be a list of floats); use 99 for eval",
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
    return parser


def raw_eval_parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Cityscapes using the 'raw' protocol"
    )
    parser = _common_parser_args(parser)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="scale of the data for evaluations",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="use HalfCityscapesEvaluator instead of CityscapesEvaluator",
    )
    return parser.parse_args()


def thin_eval_parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Cityscapes using the 'thin' protocol"
    )
    parser = _common_parser_args(parser)
    return parser.parse_args()


def evaluate(
    gt_dir: str,
    cityscapes_path: str,
    pred_path: str,
    output_path: str,
    categories: str,
    thin: bool,
    pre_seal: bool,
    nonIS: bool,
    max_dist: float,
    scale: float,
    apply_thinning: bool,
    apply_nms: bool,
    thresholds: str,
    half: bool,
    nproc: int,
):
    """Evaluate Cityscapes"""

    if categories is None:
        print("use all categories")
        categories = list(range(1, len(CityscapesEvaluator.CLASSES) + 1))
    else:
        # string evaluation for categories
        categories = categories.strip()
        try:
            categories = [int(categories)]
        except ValueError:
            try:
                if categories.startswith("[") and categories.endswith("]"):
                    categories = categories[1:-1]
                    categories = [
                        int(cat.strip()) for cat in categories.split(",")
                    ]
                else:
                    print(
                        "Bad categories format; should be a python list of floats (`[a, b, c]`)"
                    )
                    return
            except ValueError:
                print(
                    "Bad categories format; should be a python list of ints (`[a, b, c]`)"
                )
                return

    for cat in categories:
        assert (
            0 < cat < 20
        ), f"category needs to be between 1 ~ 19, but got {cat}"

    # string evaluation for thresholds
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

    if half and not thin:
        warnings.warn(
            (
                "Using Half Resolution Evaluator: using this evaluation means that the "
                "prediction size is already preprocessed to half resolution prior to running "
                "this script. If not, please don't use `--half` since it will result in "
                "unfair results."
            )
        )

    # generate output_path if given None
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_path = osp.join(
            osp.normpath(pred_path), f"edge_results_{timestamp}"
        )

    mkdir_or_exist(output_path)

    # setup logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(output_path, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    logger.info("Running Cityscapes Evaluation")
    logger.info(f"categories:         \t{categories}")
    logger.info(f"thresholds:         \t{thresholds}")
    logger.info(f"scale:              \t{scale}")
    logger.info(f"pre-seal:           \t{pre_seal}")
    logger.info(f"thinned GTs:        \t{thin}")
    logger.info(f"thinning:           \t{apply_thinning}")
    logger.info(f"nms:                \t{apply_nms}")
    logger.info(f"nonIS:              \t{nonIS}")
    logger.info(f"Half Res Evaluator: \t{half}")
    logger.info(f"GT directory:       \t{gt_dir}")
    print("\n\n")

    # get evaluator class
    evaluator_cls = HalfCityscapesEvaluator if half else CityscapesEvaluator

    # initialize evaluator
    evaluator = evaluator_cls(
        dataset_root=cityscapes_path,
        pred_root=pred_path,
        thin=thin,
        gt_dir=gt_dir,  # NOTE: we can change the directory where the preprocessed GTs are
    )
    if evaluator.sample_names is None:
        # load custom sample names
        evaluator.set_sample_names()

    # set parameters
    # evaluator.set_pred_suffix("_leftImg8bit.png")  # potato save them using .png
    eval_mode = "pre-seal" if pre_seal else "post-seal"
    instance_sensitive = not nonIS
    evaluator.set_eval_params(
        eval_mode=eval_mode,
        scale=scale,
        apply_thinning=apply_thinning,
        apply_nms=apply_nms,
        max_dist=max_dist,
        instance_sensitive=instance_sensitive,
    )

    # evaluate
    evaluator.evaluate(
        categories=categories,
        thresholds=thresholds,
        nproc=nproc,
        save_dir=output_path,
    )


def evaluate_cityscapes_raw(gt_dir: str = "gtEval"):
    args = raw_eval_parse_args()
    evaluate(
        gt_dir=gt_dir,
        cityscapes_path=args.cityscapes_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        categories=args.categories,
        thin=False,
        pre_seal=args.pre_seal,
        nonIS=args.nonIS,
        max_dist=args.max_dist,
        scale=args.scale,
        apply_thinning=False,
        apply_nms=args.apply_nms,
        thresholds=args.thresholds,
        half=args.half,
        nproc=args.nproc,
    )


def evaluate_cityscapes_thin(gt_dir: str = "gtEval"):
    args = thin_eval_parse_args()
    evaluate(
        gt_dir=gt_dir,
        cityscapes_path=args.cityscapes_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        categories=args.categories,
        thin=True,
        pre_seal=args.pre_seal,
        nonIS=args.nonIS,
        max_dist=args.max_dist,
        scale=0.5,  # doesn't matter since we're using half
        apply_thinning=True,  # apply thinning on preds
        apply_nms=args.apply_nms,
        thresholds=args.thresholds,
        half=True,  # use HalfCityscapesEvaluator
        nproc=args.nproc,
    )
