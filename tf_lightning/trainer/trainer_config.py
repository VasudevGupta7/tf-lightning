# __author__ = 'Vasudev Gupta'


class TrainerConfig(object):
    # there is no reason of keeping this. I am collecting all the possible args for `Trainer` initialization

    default_attrs = [
        'fast_dev_run', 'start_epoch', 'epochs', 'load_dir',
        'assert_consumed', 'checkpoint', 'save_only_final_ckpts',
        'save_every_ckpt', 'saved_ckpts_dir', 'max_ckpt_to_keep', 'project_name',
        'config', 'dir', 'sync_tensorboard', 'save_code', 'lit_logger',
        'keep_checkpoint_every_n_hours', 'callbacks', 'policy_name',
        'enable_precision_training', 'enable_distributed_training',
    ]
