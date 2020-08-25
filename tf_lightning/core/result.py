"""TrainResult and ValResult Class

@author: vasudevgupta
"""
from typing import Dict


class Result(Dict):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def log_dict(self, logs):
        self.logs = logs

    def log(self, **kwargs):
        self.logs = kwargs

    def get_log(self):

        logs = None

        if hasattr(self, 'log') or hasattr(self, 'log_dict'):
            logs = self.logs

        return logs


class TrainResult(Result):

    def __init__(self, loss, trainable_variables):

        super().__init__(loss=loss,
                         trainable_variables=trainable_variables)


class EvalResult(Result):

    def __init__(self, loss):

        super().__init__(loss=loss)
