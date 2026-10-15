import pandas as pd

from .BaseReporter import BaseReporter


class GenderReporter(BaseReporter):

    evaluation_type = "gender"

    def create_table(self, dataset_name=None, model_name=None):

        data = self.get_results(
            dataset_name=dataset_name,
            model_name=model_name,
        )

        rows = []

        for _, row in data.iterrows():

            rows.append({
                "Dataset": row["dataset_name"],
                "Model": row["model_name"],
                "Samples": int(row["num_samples"]),

                "Masculine score":
                    f"{row['mean_masc_score']:.4f} ± "
                    f"{row['std_masc_score']:.4f}",

                "Feminine score":
                    f"{row['mean_fem_score']:.4f} ± "
                    f"{row['std_fem_score']:.4f}",

                "Score difference":
                    f"{row['mean_score_diff']:.4f} ± "
                    f"{row['std_score_diff']:.4f}",

                "Absolute difference":
                    f"{row['mean_abs_score_diff']:.4f} ± "
                    f"{row['std_abs_score_diff']:.4f}",

                "Mean score":
                    f"{row['mean_score_mean']:.4f} ± "
                    f"{row['std_score_mean']:.4f}",

                "Masc preferred (%)":
                    round(row["masc_preferred_percentage"], 2),

                "Fem preferred (%)":
                    round(row["fem_preferred_percentage"], 2),

                "Tie (%)":
                    round(row["tie_percentage"], 2),
            })

        return pd.DataFrame(rows)