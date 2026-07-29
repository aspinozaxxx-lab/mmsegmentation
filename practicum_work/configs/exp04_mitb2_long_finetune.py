"""Experiment 4B fallback: low-LR fine-tuning from the best MiT-B2 model."""

_base_ = ['./exp03_mitb2_strong_aug.py']

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=3000, val_interval=250)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=100,
        end=3000,
        by_epoch=False)
]
optim_wrapper = dict(
    optimizer=dict(lr=0.00002))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='06_exp_mitb2_long_finetune',
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False),
        artifact_suffix=('.py', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/06_exp_mitb2_long_finetune'
