# __author__ = 'Vasudev Gupta'

import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Checkpointer(object):

    def respond_ckpt_existence(self):

        # if no ckpt is given, then nothing will be saved
        if not checkpoint:
            save_every_ckpt = False
            save_only_final_ckpt = False
            logger.info('No ckpt will be saved')

        if self.save_only_final_ckpt:
            logger.info('saving only final ckpts on training completion')
            save_every_ckpt = False

    def _get_checkpoint_manager(self, checkpoint):
        manager = tf.train.CheckpointManager(checkpoint,
                                             directory=Path(
                                                 self.lightning_base_dir, self.ckpt_dir),
                                             max_to_keep=self.max_ckpt_to_keep,
                                             keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)
        return manager

    def load_from_checkpoint(self,
                             checkpoint,
                             ckpt_name: str,
                             assert_consumed: bool):

        status = checkpoint.restore(ckpt_name)
        logger.info('ckpt_restored')

        if assert_consumed:
            status.assert_consumed()
