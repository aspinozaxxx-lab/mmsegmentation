"""Experiment 1: add multiclass Dice loss to the MiT-B0 baseline."""

_base_ = ['./baseline_segformer_mitb0.py']

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                loss_weight=1.0,
                ignore_index=255)
        ]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='03_exp_mitb0_ce_dice',
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False),
        artifact_suffix=('.py', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/03_exp_mitb0_ce_dice'
