import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from torchvision import datasets
from PIL import Image


class CustomDataset(BaseDataset):

    def __init__(self, vis_processor, text_processor, img_folder, classes=None, **kwargs):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = datasets.ImageFolder(img_folder)
        self.idx_to_class = {v: k for k, v in self.inner_dataset.class_to_idx.items()}

        if classes:
            self.classnames = classes   # user defined classes
        else:
            self.classnames = self.inner_dataset.classes  # classes parsed from folder

        self.annotation = [
            {
                "image": elem[0],
                "caption": self.idx_to_class[elem[1]],
                "label": elem[1],
                "image_id": elem[0]
            }
            for elem in self.inner_dataset.imgs
            if self.idx_to_class[elem[1]] in classes
        ]

        self._add_instance_ids()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }

    def display_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "label": self.idx_to_class[ann["label"]],
                "image": sample["image"],
            }
        )


class CustomEvalDataset(CustomDataset):
    def __init__(self, vis_processor, text_processor, img_folder):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, img_folder=img_folder)

        self.text = [self.text_processor(i) for i in self.classnames]
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            caption = ann["caption"]
            self.img2txt[img_id].append(self.inner_dataset.class_to_idx[caption])

    def __getitem__(self, index):

        image_path = self.annotation[index]["image"]
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}
