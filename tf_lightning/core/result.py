# __author__ = 'Vasudev Gupta'

class Result(dict):
    # parent class of TrainResult & EvalResult

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
    """[Compulsary]
    Just use this class at the end of your training step..
    Both arguments are compulsary

    Args:
        minimize: pass the loss, you want to minimize
        trainable_variables: pass the variables w.r.t which you want to find gradients
    """

    def __init__(self, minimize, trainable_variables):

        super().__init__(minimize=minimize,
                         trainable_variables=trainable_variables)


class EvalResult(Result):
    """[Optional]
    Whatever args you will be passing here, will be passed as args in lit-callbacks method
    Args:
        minimize: You can simply pass the loss, if you wish..
    """

    def __init__(self, minimize):

        super().__init__(minimize=minimize)
