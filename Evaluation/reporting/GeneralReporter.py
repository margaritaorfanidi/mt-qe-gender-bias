import pandas as pd

from .BaseReporter import BaseReporter


class GeneralReporter(BaseReporter):

    evaluation_type = "general"

    def create_table(self, dataset_name=None, model_name=None):

        data = self.get_results(
            dataset_name=dataset_name,
            model_name=model_name,
        )

        rows = []

        for _, row in data.iterrows():

            table_row = {
                "Dataset": row["dataset_name"],
                "Model": row["model_name"],
                "Samples": int(row["num_samples"]),

                "Predicted score":
                    f"{row['mean_predicted_score']:.4f} ± "
                    f"{row['std_predicted_score']:.4f}",
            }

            if pd.notna(row.get("mean_target_score")):

                table_row["Target score"] = (
                    f"{row['mean_target_score']:.4f} ± "
                    f"{row['std_target_score']:.4f}"
                )

                table_row["Pearson"] = round(
                    row["pearson"],
                    4
                )

                table_row["Spearman"] = round(
                    row["spearman"],
                    4
                )

            rows.append(table_row)

        return pd.DataFrame(rows)