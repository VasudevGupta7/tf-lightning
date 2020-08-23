"""Callback Class

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np

import time
import logging
import wandb

logger = logging.getLogger(__name__)

class Callback(object):

    def __init__(self):
        """
        You can overwrite this class whenever you need to :)
        """
        logger.info('using in-built callbacks or its extension')
    
    def on_train_begin(self):
        logger.info('Training Inititated')

    def on_train_end(self):
        logger.info('YAAYYYY, Model is trained')

    def on_epoch_begin(self, epoch):
        
        self.start_epoch= time.time()

        logger.info(f'epoch-{epoch} started')

    def on_epoch_end(self, epoch, tr_loss, val_loss):

        # logging per epoch
        epoch_metrics= {
                'epoch': epoch,
                "epoch_tr_loss": tr_loss.numpy(),
                'epoch_val_loss': val_loss.numpy(),
            }

        wandb.log(epoch_metrics, commit= False)

        time_delta = np.around(time.time() - self.start_epoch, 2)

        print(f"EPOCH-{epoch} ===== TIME TAKEN-{time_delta}sec ===== {epoch_metrics}")

        return epoch_metrics

    def on_batch_begin(self, batch_idx):
        logger.info(f'batch-{batch_idx} started')

    def on_batch_end(self, batch_idx, tr_loss, val_loss):

        # logging per step
        step_metrics= {
                    'batch_idx': batch_idx.numpy(),
                    "batch_tr_loss": tr_loss.numpy(),
                    'batch_val_loss': val_loss.numpy(),
                }
        wandb.log(step_metrics)

        print(f"lightning-logs ===== {step_metrics}")

        return step_metrics
