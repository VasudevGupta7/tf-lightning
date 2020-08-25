"""Configuration file for Trainer

@author: vasudevgupta
"""


class TrainerConfig(object):

    default_attrs = [
        'fast_dev_run', 'start_epoch', 'epochs', 'load_dir',
        'assert_consumed', 'checkpoint', 'save_only_final_ckpts',
        'save_every_ckpt', 'saved_ckpts_dir', 'max_ckpt_to_keep', 'project_name',
        'config', 'dir', 'sync_tensorboard', 'save_code', 'lit_logger',
        'keep_checkpoint_every_n_hours', 'callbacks', 'enable_function',
    ]
