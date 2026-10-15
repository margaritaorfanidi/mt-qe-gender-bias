# for each metric i want  to evaluate i need to create a class 
# that inherits BaseQEMetric

class BaseQEMetric:

    def __init__(self, model):
        self.model = model

    def score( self, sources, translations, batch_size=32, gpus=1, progress_bar=True,):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, model_name=None):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, model_name):
        raise NotImplementedError


