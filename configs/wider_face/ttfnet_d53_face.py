# model settings
input_size = 512
model = dict(
    type='TTFNet',
    pretrained='./pretrain/darknet53.pth',
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_eval=False),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(128, 256, 512, 1024),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=2,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/widerface/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'WIDER_train/images/Annotations/train.txt',
            img_prefix=data_root + 'WIDER_train/images',
            min_size=17,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/images/Annotations/val.txt',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/images/Annotations/val.txt',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[18, 22])
checkpoint_config = dict(interval=1)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ttfnet_d53_face'
load_from = None
resume_from = "work_dirs/ttfnet_d53_face/latest.pth"
resume_from = None
workflow = [('train', 1)]
