"""Logger class for Wandb and Tensorboard

@author: vasudevgupta
"""
import wandb


class WandbLogger(object):

    def __init__(self,
                 project_name=None,
                 config=None,
                 log_dir=None,
                 sync_tensorboard=False,
                 save_code=None):

        wandb.init(project=project_name,
                   config=config,
                   dir=log_dir,
                   sync_tensorboard=sync_tensorboard,
                   save_code=save_code)

    def log(self, info, commit=True, step=None):
        wandb.log(info, commit=commit, step=step)
