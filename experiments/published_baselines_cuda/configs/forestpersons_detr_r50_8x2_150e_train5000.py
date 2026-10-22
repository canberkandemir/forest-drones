"""ForestPersons train5000 fine-tuning config for original DETR R-50.

Use with MMDetection v2.28.2 on a CUDA machine. Set MMDET_ROOT to a local
checkout of https://github.com/open-mmlab/mmdetection at tag v2.28.2.
"""

import os

_base_ = [
    os.path.join(
        os.environ.get("MMDET_ROOT", "third_party/mmdetection"),
        "configs/detr/detr_r50_8x2_150e_coco.py",
    )
]

classes = ("person",)
data_root = "data/"

model = dict(bbox_head=dict(num_classes=1))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        ann_file="outputs/baseline_feasibility/forestpersons_train5000_oneclass_coco.json",
        img_prefix=data_root + "images/",
        classes=classes,
    ),
    val=dict(
        type="CocoDataset",
        ann_file="outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json",
        img_prefix=data_root + "images/",
        classes=classes,
    ),
    test=dict(
        type="CocoDataset",
        ann_file="outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json",
        img_prefix=data_root + "images/",
        classes=classes,
    ),
)

evaluation = dict(interval=1, metric="bbox", classwise=True)
checkpoint_config = dict(interval=5)
runner = dict(type="EpochBasedRunner", max_epochs=150)

# Paper-reported DETR hyperparameters: AdamW, lr 1e-4, batch 16, weight decay
# 1e-4, 150 epochs. This config keeps the schedule but lowers single-GPU batch
# size for memory. Scale lr deliberately when changing total batch size.
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.0001)
