"""Baseline 1: FCN ResNet-50-D8 with cross-entropy losses."""

_base_ = [
    '../../configs/_base_/models/fcn_r50-d8.py',
    './_base_/dataset.py',
    './_base_/runtime.py',
]

crop_size = (256, 256)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=3, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=3, norm_cfg=norm_cfg))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='01_baseline_fcn_r50',
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False),
        artifact_suffix=('.py', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/01_baseline_fcn_r50'
