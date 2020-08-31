# __author__ = 'Vasudev Gupta'


class TrainerConfig(object):

    # training
    fast_dev_run: bool = False
    start_epoch: int = 1
    epochs: int = 10

    lightning_base_dir: str = 'lightning_stuff'

    # checkpoints loading
    load_dir: str = ''
    assert_consumed: bool = False

    # checkpoints saving
    ckpt_dir: str = 'ckpt_dir'
    save_only_final_ckpt: bool = False
    save_every_ckpt: bool = True
    max_ckpt_to_keep: int = 3
    keep_checkpoint_every_n_hours: int = None

    # precision training related stuff
    enable_precision_training: bool = False
    policy_name: str = 'mixed_float16'

    # loggers
    enable_loggers: bool = True
    loggers: list = []

    # wandb related stuff
    project_name: str = 'lightning-project'
    litmodule_config: dict = {}
    sync_tensorboard: bool = False

    # callbacks related stuff
    enable_callbacks: bool = True
    callbacks = None

    # distributed training related stuff
    enable_distributed_training: bool = False

    deafult_attrs = ['fast_dev_run', 'start_epoch', 'epochs',
                     'load_dir', 'assert_consumed', 'save_only_final_ckpt',
                     'max_ckpt_to_keep', 'keep_checkpoint_every_n_hours',
                     'enable_precision_training', 'policy_name',
                     'enable_loggers', 'loggers', 'project_name',
                     'litmodule_config', 'sync_tensorboard',
                     'enable_callbacks', 'callbacks', 'enable_distributed_training']
