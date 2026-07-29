"""Experiment 4A: validation/test-time augmentation for the MiT-B2 model."""

_base_ = ['./exp03_mitb2_strong_aug.py']

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='06_exp_mitb2_tta'),
        artifact_suffix=('.py', '.pth', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/06_exp_mitb2_tta'
