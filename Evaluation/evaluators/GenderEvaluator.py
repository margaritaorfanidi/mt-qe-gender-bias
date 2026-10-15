from .BaseEvaluator  import BaseEvaluator

class GenderEvaluator(BaseEvaluator):

    def evaluate(self):

        print("Scoring masculine translations...")

        masc_scores = self.metric.score(
            self.dataset.sources,
            self.dataset.masculine_translations,
            batch_size=self.batch_size,
            gpus=self.gpus,
            progress_bar=self.progress_bar,
        )

        print("Scoring feminine translations...")

        fem_scores = self.metric.score(
            self.dataset.sources,
            self.dataset.feminine_translations,
            batch_size=self.batch_size,
            gpus=self.gpus,
            progress_bar=self.progress_bar,
        )

        result = self.dataset.dataframe.copy()

        result["score_masc"] = masc_scores
        result["score_fem"] = fem_scores

        return result
