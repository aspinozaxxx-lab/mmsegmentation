"""Experiment 2: stronger geometric and photometric augmentations."""

_base_ = ['./exp01_mitb0_ce_dice.py']

crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(320, 320),
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='04_exp_mitb0_strong_aug'),
        artifact_suffix=('.py', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/04_exp_mitb0_strong_aug'
