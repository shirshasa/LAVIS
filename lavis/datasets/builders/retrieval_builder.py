"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import warnings

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.custom_dataset import CustomDataset, CustomEvalDataset
from lavis.datasets.datasets.retrieval_datasets import (
    RetrievalDataset,
    RetrievalEvalDataset,
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_ret.yaml"}


@registry.register_builder("custom_retrieval")
class CustomRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = CustomDataset
    eval_dataset_cls = CustomEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/custom_clf_dataset.yaml"}

    def _download_data(self):
        pass

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        classes = self.config.classes

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            assert isinstance(ann_paths, str)

            if not os.path.exists(ann_paths):
                warnings.warn("storage path {} does not exist.".format(ann_paths))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                img_folder=ann_paths,
                classes=classes
            )

        return datasets


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults.yaml"}
