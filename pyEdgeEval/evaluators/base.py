#!/usr/bin/env python3

import json
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from prettytable import PrettyTable

from pyEdgeEval.common.multi_label import (
    save_overall_metric,
    save_pretty_metrics,
)
from pyEdgeEval.utils import print_log


class BaseEvaluator(object, metaclass=ABCMeta):

    # Global variables
    dataset_root = None
    pred_root = None
    split = None

    # Hidden variables
    _sample_names = None  # don't make this mutable (e.g. [])
    _logger = "pyEdgeEval"

    @property
    def sample_names(self):
        if self._sample_names is None:
            print_log("sample_names is None", logger=self._logger)
            return None
        else:
            # avoid changing inplace
            return deepcopy(self._sample_names)

    @abstractmethod
    def set_sample_names(self, **kwargs):
        """Placeholder for setting sample_names"""
        pass

    @abstractmethod
    def set_eval_params(self, **kwargs):
        """Placeholder for setting evaluation parameters"""
        pass

    @abstractmethod
    def eval_params(self):
        """Placeholder for getting evaluation parameters"""
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        """Placeholder for the main evaluation function"""
        pass


class BaseBinaryEvaluator(BaseEvaluator):
    ...


class BaseMultilabelEvaluator(BaseEvaluator):

    CLASSES = None

    @abstractmethod
    def _before_evaluation(self):
        pass

    @abstractmethod
    def evaluate_category(self, **kwargs):
        """Placeholder for the category evaluation function"""
        pass

    def evaluate(
        self,
        categories,
        thresholds,
        nproc,
        save_dir,
        print_metrics=(
            "ODS_threshold",
            "ODS_recall",
            "ODS_precision",
            "ODS_f1",
            "AUC",
        ),
    ):
        self._before_evaluation()

        # print evaluation params
        pretty_eval_params = json.dumps(
            self.eval_params, sort_keys=False, indent=4
        )
        print_log(pretty_eval_params, logger=self._logger)

        # check number of categories (indexed from 1)
        if isinstance(categories, int):
            categories = [categories]
        if categories is None:
            categories = list(range(1, len(self.CLASSES) + 1))
        assert isinstance(
            categories, list
        ), f"ERR: `categories` should be a list, but got {type(categories)}"
        assert len(categories) > 0, "ERR: 0 categories"

        ret_metrics = OrderedDict()
        for category in categories:
            # do a single category evaluation
            overall_metric = self.evaluate_category(
                category=category,
                thresholds=thresholds,
                nproc=nproc,
                save_dir=save_dir,
            )
            for k, v in overall_metric.items():
                if k in ret_metrics.keys():
                    ret_metrics[k].append(v)
                else:
                    ret_metrics[k] = [v]

        if save_dir:
            # output basic metrics (unformatted)
            ret_metrics_out = OrderedDict(
                {
                    ret_metric: np.round(np.nanmean(ret_metric_value), 6)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                }
            )
            save_overall_metric(
                root_dir=save_dir,
                overall_metric=ret_metrics_out,
                file_name="overall_eval_bdry.txt",
            )

        # isolate metrics that are not percentage
        ret_thresholds = ret_metrics.pop("ODS_threshold", None)

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 3)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        # each class table
        remove_metrics = ["OIS_recall", "OIS_precision", "OIS_f1", "AUC", "AP"]
        for m in remove_metrics:
            ret_metrics.pop(m, None)

        class_names = [self.CLASSES[i - 1] for i in categories]
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(np.array(ret_metric_value) * 100, 3)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update(
            {"ODS_threshold": np.round(np.array(ret_thresholds), 2)}
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if key in print_metrics or key == "Class":
                class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key in print_metrics:
                summary_table_data.add_column(key, [val])

        print_log("per class results:", logger=self._logger)
        print_log("\n" + class_table_data.get_string(), logger=self._logger)
        print_log("Summary:", logger=self._logger)
        print_log("\n" + summary_table_data.get_string(), logger=self._logger)

        # save total metrics
        if save_dir:
            save_pretty_metrics(
                root_dir=save_dir,
                class_table=class_table_data.get_string(),
                summary_table=summary_table_data.get_string(),
            )
