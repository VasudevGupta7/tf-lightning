# __author__ = 'Vasudev Gupta'

# Overwrite this class whenever you need

import logging

logger = logging.getLogger(__name__)


class Callback(object):
    """
    Methods:
        on_train_begin: This is getting called when training is started
        on_train_end: This is getting called when training is ended

        on_epoch_begin: This is getting called whenever each epoch is started
        on_epoch_end: This is getting called whenever each epoch is ended

        on_batch_begin: This is getting called whenever each batch is started
        on_batch_end: This is getting called whenever each batch is ended
    """

    def __init__(self):
        pass

    def on_train_begin(self):
        logger.info('Training Inititated')

    def on_train_end(self):
        logger.info('YAAYYYY, Model is trained')

    def on_epoch_begin(self, epoch):
        logger.info(f'epoch-{epoch} started')

    def on_epoch_end(self, epoch):
        logger.info(f'epcoh-{epoch} ended')

    def on_batch_begin(self, batch_idx):
        logger.info(f'batch-{batch_idx} started')

    def on_batch_end(self, batch_idx, tr_result, val_result):
        """
        Args:
            tr_result: Its the same thing which you are returning in `TrainResult`
            val_result: Its the same thing which you are returning in `EvalResult`
        """
        logger.info(f'batch-{batch_idx} ended')
