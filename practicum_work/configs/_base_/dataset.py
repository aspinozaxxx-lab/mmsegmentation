"""Dataset and dataloader settings shared by all Sprint 6 experiments."""

dataset_type = 'BaseSegDataset'
data_root = 'train_dataset_cleaned'
crop_size = (256, 256)
metainfo = dict(
    classes=('background', 'cat', 'dog'),
    palette=[(35, 35, 35), (240, 70, 70), (65, 135, 245)])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=scale, keep_ratio=False)
                for scale in ((192, 192), (256, 256), (320, 320))
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations', reduce_zero_label=False)],
            [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/train', seg_map_path='labels/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/test', seg_map_path='labels/test'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline))

val_evaluator = dict(
    type='IoUMetric', ignore_index=255, iou_metrics=['mDice', 'mIoU'])
test_evaluator = dict(
    type='IoUMetric', ignore_index=255, iou_metrics=['mDice', 'mIoU'])
