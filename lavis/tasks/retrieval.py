"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
            )
            logging.info(eval_result)

            self._report_classification_results(
                score_i2t, data_loader.dataset.img2txt, data_loader.dataset.classnames
            )

        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr2 = 100.0 * len(np.where(ranks < 2)[0]) / len(ranks)
        tr3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)

        # Text->Images
        # ranks = np.zeros(scores_t2i.shape[0])
        #
        # for index, score in enumerate(scores_t2i):
        #     inds = np.argsort(score)[::-1]
        #     ranks[index] = np.where(inds == txt2img[index])[0][0]
        #
        # # Compute metrics
        # ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        # ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        # ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        #
        tr_mean = (tr1 + tr2 + tr3) / 3
        # ir_mean = (ir1 + ir5 + ir10) / 3
        # r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr2 + tr3) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr2,
            "txt_r10": tr3,
            "txt_r_mean": tr_mean,
            # "img_r1": ir1,
            # "img_r5": ir5,
            # "img_r10": ir10,
            # "img_r_mean": ir_mean,
            # "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result

    @torch.no_grad()
    def _report_classification_results(self, scores_i2t, img2txt, classes):
        y_pred = []
        y_true = []

        for index, score in enumerate(scores_i2t):
            ind_max_score = np.argmax(score)
            true_text_ind = img2txt[index][0]  # TODO
            y_pred.append(classes[ind_max_score])
            y_true.append(classes[true_text_ind])

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)

        clf_report = classification_report(y_pred=y_pred, y_true=y_true)

        logging.info(f"F1 score: {fscore:.2f}")
        logging.info(f"Accuracy: {accuracy:.2f}")
        logging.info(f"\n{clf_report}\n")
        logging.info(f"\n{confusion_matrix(y_pred=y_pred, y_true=y_true)}\n")

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(f"F1 score: {fscore:.2f}" + "\n")
            f.write(f"Accuracy: {accuracy:.2f}")
            f.write(f"\n{clf_report}\n")
            f.write(f"\n{confusion_matrix(y_pred=y_pred, y_true=y_true)}\n")
