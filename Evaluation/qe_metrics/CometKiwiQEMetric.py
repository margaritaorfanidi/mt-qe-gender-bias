from comet import (
    download_model,
    load_from_checkpoint as comet_load_from_checkpoint,
)
from comet.models import UnifiedMetric

from .BaseQEMetric import BaseQEMetric


class CometKiwiQEMetric(BaseQEMetric):

    DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"

    @classmethod
    def load_from_checkpoint(cls, model_path):

        model = UnifiedMetric.load_from_checkpoint(
            model_path,
            load_pretrained_weights=False,
            map_location="cpu",
        )
        model.eval()

        return cls(model)

    @classmethod
    def load_from_pretrained(cls, model_name=None):

        if model_name is None:
            model_name = cls.DEFAULT_MODEL

        model_path = download_model(model_name)

        model = comet_load_from_checkpoint(model_path)
        model.eval()

        return cls(model)

    def score(
        self,
        sources,
        translations,
        batch_size=32,
        gpus=1,
        progress_bar=True,
    ):

        inputs = [
            {
                "src": src,
                "mt": mt,
            }
            for src, mt in zip(sources, translations)
        ]

        output = self.model.predict(
            inputs,
            batch_size=batch_size,
            gpus=gpus,
            progress_bar=progress_bar,
        )

        return output.scores