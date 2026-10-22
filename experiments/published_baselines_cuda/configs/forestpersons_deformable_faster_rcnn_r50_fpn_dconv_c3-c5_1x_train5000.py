"""ForestPersons train5000 fine-tuning config for Deformable Faster R-CNN.

Use with MMDetection v2.28.2 on a CUDA machine. Set MMDET_ROOT to a local
checkout of https://github.com/open-mmlab/mmdetection at tag v2.28.2.
"""

import os

_base_ = [
    os.path.join(
        os.environ.get("MMDET_ROOT", "third_party/mmdetection"),
        "configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py",
    )
]

classes = ("person",)
data_root = "data/"

model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

data = dict(
    samples_per_gpu=2,
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
checkpoint_config = dict(interval=1)
runner = dict(type="EpochBasedRunner", max_epochs=12)

# Paper-reported Deformable R-CNN hyperparameters: SGD, lr 2e-2, batch 16,
# weight decay 1e-4, 12 epochs. This config uses a smaller per-GPU batch for
# single-GPU smoke/fine-tuning; scale lr if changing total batch size.
optimizer = dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001)
