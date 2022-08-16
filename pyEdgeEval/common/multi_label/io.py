#!/usr/bin/env python3

import os.path as osp

from pyEdgeEval.utils import mkdir_or_exist


def save_sample_metrics(
    root_dir: str, sample_metrics, file_name: str = "eval_bdry_img.txt"
):
    file_path = osp.join(root_dir, file_name)
    tmp_line = "{name} {thrs:<10.6f} {rec:<10.6f} {prec:<10.6f} {f1:<10.6f}\n"
    with open(file_path, "w") as f:
        for res in sample_metrics:
            f.write(
                tmp_line.format(
                    name=res["name"],
                    thrs=res["threshold"],
                    rec=res["recall"],
                    prec=res["precision"],
                    f1=res["f1"],
                )
            )


def save_threshold_metrics(
    root_dir: str, threshold_metrics, file_name: str = "eval_bdry_thr.txt"
):
    file_path = osp.join(root_dir, file_name)
    tmp_line = "{thrs:<10.6f} {rec:<10.6f} {prec:<10.6f} {f1:<10.6f}\n"
    with open(file_path, "w") as f:
        for res in threshold_metrics:
            f.write(
                tmp_line.format(
                    thrs=res["threshold"],
                    rec=res["recall"],
                    prec=res["precision"],
                    f1=res["f1"],
                )
            )


def save_overall_metric(
    root_dir: str,
    overall_metric,
    file_name: str = "eval_bdry.txt",
):
    file_path = osp.join(root_dir, file_name)
    with open(file_path, "w") as f:
        f.write(
            "{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                overall_metric["ODS_threshold"],
                overall_metric["ODS_recall"],
                overall_metric["ODS_precision"],
                overall_metric["ODS_f1"],
                overall_metric["OIS_recall"],
                overall_metric["OIS_precision"],
                overall_metric["OIS_f1"],
                overall_metric["AUC"],
                overall_metric["AP"],
            )
        )


def save_pretty_metrics(
    root_dir: str,
    class_table,
    summary_table,
    file_name: str = "results.txt",
):
    file_path = osp.join(root_dir, file_name)
    data = (
        "Per Class Results:\n"
        + class_table
        + "\n"
        + "Summary:\n"
        + summary_table
    )
    with open(file_path, "w") as f:
        f.write(data)


def save_category_results(
    root: str,
    category: int,
    sample_metrics,
    threshold_metrics,
    overall_metric,
):
    """Save per-category results"""
    cat_name = "class_" + str(category).zfill(3)
    cat_dir = osp.join(root, cat_name)
    mkdir_or_exist(cat_dir)

    if sample_metrics:
        save_sample_metrics(cat_dir, sample_metrics)
    if threshold_metrics:
        save_threshold_metrics(cat_dir, threshold_metrics)
    if overall_metric:
        save_overall_metric(cat_dir, overall_metric)
