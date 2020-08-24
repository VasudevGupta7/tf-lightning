"""TrainResult and ValResult Class

@author: vasudevgupta
"""

class TrainResult(dict):

    def __init__(self, loss, trainable_variables, logs):
        self.log=None
        super().__init__(loss=loss,
                         trainable_variables=trainable_variables,
                         log=self.log)

class ValResult(dict):

    def __init__(self, loss):
        self.log=None
        super().__init__(loss=loss,
                         log=self.log)
