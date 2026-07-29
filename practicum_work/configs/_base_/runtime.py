"""Reproducible iteration-based runtime shared by training experiments."""

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook', draw=False, interval=1))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mDice',
        rule='greater',
        min_delta=0.1,
        patience=8,
        strict=True)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=6000, val_interval=250)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=200,
        end=6000,
        by_epoch=False)
]

log_processor = dict(by_epoch=False, window_size=20)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'
randomness = dict(seed=42, deterministic=True)

tta_model = dict(type='SegTTAModel')
