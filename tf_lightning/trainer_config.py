"""Configuration file for Trainer

@author: vasudevgupta
"""
from tf_lightning.callbacks import Callback
from tf_lightning.loggers import WandbLogger

class TrainerConfig(object):
    
    ## default args
        
    # running on only 1 batch for only 1 epoch, No ckpts will be saved
    fast_dev_run = False
        
    start_epoch = 1
    epochs = 10
        
    # if specifying load_dir, it will load_ckpt else it won't load ckpt
    load_dir = ''
    # if loading ckpt, use this ard to check whether all saved objects are restored
    assert_consumed = False
        
    # You need to specify ckpt, if you want to use other save options
    checkpoint = None
        
    # Note: these options will be affected only if you specify checkpoint
    save_only_final_ckpts = False
    save_every_ckpt = False
        
    # by default, all tf_lightning related stuff will be saved in this dir
    lightning_base_dir = 'lightning_stuff'
    ckpt_dir = 'ckpts'
    log_dir= 'logs'
        
    # Arguements related to ckpts
    max_ckpt_to_keep = 3
    keep_checkpoint_every_n_hours = None
    
    # these arguments are valid only for defalt wandb
    # wandb related arguments
    project_name = 'tf-lightning-project'
    config = None
    sync_tensorboard = False
    save_code = None
    
    # Wandb is supported by default
    lit_logger = WandbLogger(project_name=project_name,
                   config=config,
                   sync_tensorboard=sync_tensorboard,
                   save_code=save_code)

    # You can override Callback class and customize methods
    callbacks= Callback()
        
    # if bool(load_dir):
    #     assert(Save)
    
    default_attrs= [
            'fast_dev_run', 'start_epoch', 'epochs', 'load_dir',
            'assert_consumed', 'checkpoint', 'save_only_final_ckpts',
            'save_every_ckpt', 'saved_ckpts_dir', 'max_ckpt_to_keep', 'project_name',
            'config', 'dir', 'sync_tensorboard', 'save_code', 'lit_logger',
            'keep_checkpoint_every_n_hours', 'callbacks', 'enable_function',
                    ]
    
    def __init__(self):
        pass

    def get_config(self):
        return self.default_attrs
