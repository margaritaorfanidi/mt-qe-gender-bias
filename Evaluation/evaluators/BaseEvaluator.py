class BaseEvaluator:

    def __init__(
        self,
        metric,
        dataset,
        batch_size=32,
        gpus=1,
        progress_bar=True,
    ):
        self.metric = metric
        self.dataset = dataset
        self.batch_size = batch_size
        self.gpus = gpus
        self.progress_bar = progress_bar

    def evaluate(self):
        raise NotImplementedError
        