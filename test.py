# __author__ = 'Vasudev Gupta'

import tf_lightning as tl
import tensorflow as tf


class TestModel(tl.LightningModule):
    # just a random model with random dataset

    def __init__(self):
        # simple test model
        super().__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(5),
            tf.keras.layers.Dense(2)
        ])

    def call(self, dataset):
        return self.model(dataset)

    def configure_optimizers(self):
        return tf.keras.optimizers.Adam(0.1),

    def training_step(self, batch, batch_idx, optimizer_idx):

        pred = self(batch)
        loss = tf.reduce_mean(pred)

        log = {'batch_idx': batch_idx, 'tr_loss': loss}
        result = tl.TrainResult(
            loss, self.model.trainable_variables, log=log)

        return result

    def validation_step(self, batch, batch_idx, optimizer_idx):

        pred = self(batch)
        loss = tf.reduce_mean(pred)

        log = {'batch_idx': batch_idx, 'val_loss': loss}
        result = tl.EvalResult(loss, log=log)

        return result

    def checkpointer(self):
        return tf.train.Checkpoint(m=self.model,
                                   opt0=self.optimizer_0)


class TestDataLoader(tl.LightningDataModule):
    # using random dataset

    def __init__(self):
        self.batch_size = 32

    def setup(self):
        self.tr_dataset = tf.random.normal((256, 7))
        self.val_dataset = tf.random.normal((64, 7))

    def train_dataloader(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            self.tr_dataset).batch(self.batch_size)
        return dataset

    def val_dataloader(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            self.val_dataset).batch(self.batch_size)
        return dataset


if __name__ == '__main__':

    model = TestModel()

    dataloader = TestDataLoader()

    trainer = tl.Trainer()

    trainer.fit(model, dataloader)
