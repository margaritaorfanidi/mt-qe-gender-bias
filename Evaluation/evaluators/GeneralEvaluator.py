
from .BaseEvaluator import BaseEvaluator

class GeneralEvaluator(BaseEvaluator):

    def evaluate(self):

        print("Scoring translations...")

        scores = self.metric.score(
            self.dataset.sources,
            self.dataset.translations,
            batch_size=self.batch_size,
            gpus=self.gpus,
            progress_bar=self.progress_bar,
        )

        result = self.dataset.dataframe.copy()

        result["predicted_score"] = scores

        return result

