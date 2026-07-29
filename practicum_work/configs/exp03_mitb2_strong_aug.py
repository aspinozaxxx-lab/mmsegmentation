"""Experiment 3: replace MiT-B0 with the larger MiT-B2 backbone."""

_base_ = ['./exp02_mitb0_strong_aug.py']

checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b2_20220624-66e8bf70.pth')
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='05_exp_mitb2_strong_aug'),
        artifact_suffix=('.py', '.pth', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/05_exp_mitb2_strong_aug'
