"""Baseline 2: SegFormer MiT-B0 with cross-entropy loss."""

_base_ = [
    '../../configs/_base_/models/segformer_mit-b0.py',
    './_base_/dataset.py',
    './_base_/runtime.py',
]

crop_size = (256, 256)
checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b0_20220624-7e0fe6dd.pth')
model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        num_classes=3, norm_cfg=dict(type='BN', requires_grad=True)))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        }))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='02_baseline_segformer_mitb0'),
        artifact_suffix=('.py', '.pth', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/02_baseline_segformer_mitb0'
