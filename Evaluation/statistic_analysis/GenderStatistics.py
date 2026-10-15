

import numpy as np

from .BaseStatistics import BaseStatistics




class GenderStatistics(BaseStatistics):

    @staticmethod
    def compute_scores(masc_scores, fem_scores):

        if len(masc_scores) != len(fem_scores):
            raise ValueError("masc_scores and fem_scores must have the same length.")

        masc_scores = np.array(masc_scores, dtype=float)

        fem_scores = np.array(fem_scores, dtype=float)

        score_diff = masc_scores - fem_scores

        return {
            "score_masc": masc_scores,
            "score_fem": fem_scores,
            "score_diff": score_diff,
            "abs_score_diff": np.abs(score_diff),
            "score_mean": (masc_scores + fem_scores) / 2,
                }


    @staticmethod
    def summarize(score_data):

        masc_scores = score_data["score_masc"]
        fem_scores = score_data["score_fem"]
        score_diff = score_data["score_diff"]
        abs_score_diff = score_data["abs_score_diff"]
        score_mean = score_data["score_mean"]

        return {
            "num_samples": int(len(masc_scores)),

            # Masculine scores
            "mean_masc_score": float(np.mean(masc_scores)),
            "std_masc_score": float(np.std(masc_scores)),
            "median_masc_score": float(np.median(masc_scores)),
            "min_masc_score": float(np.min(masc_scores)),
            "max_masc_score": float(np.max(masc_scores)),

            # Feminine scores
            "mean_fem_score": float(np.mean(fem_scores)),
            "std_fem_score": float(np.std(fem_scores)),
            "median_fem_score": float(np.median(fem_scores)),
            "min_fem_score": float(np.min(fem_scores)),
            "max_fem_score": float(np.max(fem_scores)),

            # Signed difference
            "mean_score_diff": float(np.mean(score_diff)),
            "std_score_diff": float(np.std(score_diff)),
            "median_score_diff": float(np.median(score_diff)),
            "min_score_diff": float(np.min(score_diff)),
            "max_score_diff": float(np.max(score_diff)),

            # Absolute gap
            "mean_abs_score_diff": float(np.mean(abs_score_diff)),
            "std_abs_score_diff": float(np.std(abs_score_diff)
            ),
            "median_abs_score_diff": float(np.median(abs_score_diff)),
            "max_abs_score_diff": float(np.max(abs_score_diff)),

            # Average pair score
            "mean_score_mean": float(np.mean(score_mean)),
            "std_score_mean": float(np.std(score_mean)),
            "median_score_mean": float(np.median(score_mean)),

            # Preferences
            "masc_preferred_count": int(np.sum(score_diff > 0)),
            "fem_preferred_count": int(np.sum(score_diff < 0)),
            "tie_count": int(np.sum(score_diff == 0)),

            "masc_preferred_percentage": float(np.mean(score_diff > 0) * 100),
            "fem_preferred_percentage": float(np.mean(score_diff < 0) * 100),
            "tie_percentage": float(np.mean(score_diff == 0) * 100),
        }
