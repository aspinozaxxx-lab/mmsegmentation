"""Final one-shot test task, using the checkpoint selected on validation."""

_base_ = ['./exp03_mitb2_strong_aug.py']

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='Practicum/Sprint6-mmsegmentation',
            task_name='07_final_test_selected_model'),
        artifact_suffix=('.py', '.json', '.csv'))
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.6)
work_dir = 'work_dirs/practicum/07_final_test_selected_model'
