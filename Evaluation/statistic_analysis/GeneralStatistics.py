import numpy as np
from scipy.stats import pearsonr, spearmanr

from .BaseStatistics import BaseStatistics


class GeneralStatistics(BaseStatistics):

    @staticmethod
    def compute_scores(predicted_scores, target_scores=None):

        predicted_scores = np.array(predicted_scores, dtype=float)

        score_data = {
            "predicted_score": predicted_scores
        }

        if target_scores is not None:

            if len(predicted_scores) != len(target_scores):
                raise ValueError(
                    "predicted_scores and target_scores must have the same length."
                )

            score_data["target_score"] = np.array(
                target_scores,
                dtype=float
            )

        return score_data


    @staticmethod
    def summarize(score_data):

        predicted_scores = score_data["predicted_score"]

        summary = {
            "num_samples": int(len(predicted_scores)),
            "mean_predicted_score": float(np.mean(predicted_scores)),
            "std_predicted_score": float(np.std(predicted_scores)),
            "median_predicted_score": float(np.median(predicted_scores)),
            "min_predicted_score": float(np.min(predicted_scores)),
            "max_predicted_score": float(np.max(predicted_scores)),
        }

        if "target_score" in score_data:

            target_scores = score_data["target_score"]

            summary.update({
                "mean_target_score": float(np.mean(target_scores)),
                "std_target_score": float(np.std(target_scores)),
                "median_target_score": float(np.median(target_scores)),
                "min_target_score": float(np.min(target_scores)),
                "max_target_score": float(np.max(target_scores)),
                "pearson": float(pearsonr(target_scores, predicted_scores).statistic),
                "spearman": float(spearmanr(target_scores, predicted_scores).statistic),
            })

        return summary