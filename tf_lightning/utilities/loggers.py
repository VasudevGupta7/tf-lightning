"""Logger class for Wandb and Tensorboard

@author: vasudevgupta
"""
import wandb


class WandbLogger(object):

    def __init__(self,
                 project_name,
                 config,
                 ):

        wandb.init(project=project_name,
                   config=config
                   
        )

    def log(self, info, commit=True, step=None):
        wandb.log(info, commit=commit, step=step)
