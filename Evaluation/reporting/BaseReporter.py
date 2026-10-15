import json
from pathlib import Path

import pandas as pd


class BaseReporter:

    evaluation_type = None

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results = None

    def load_results(self):

        rows = []
        # for each file 
        for file in self.results_dir.glob("*.json"):

            with open(file, "r", encoding="utf-8") as f: result = json.load(f)

            if result.get("evaluation_type") != self.evaluation_type: continue

            row = {
                "experiment_name": result.get("experiment_name"),
                "model_name": result.get("model_name"),
                "dataset_name": result.get("dataset_name"),
            }

            row.update(result.get("statistics", {}))

            rows.append(row)

        self.results = pd.DataFrame(rows)

        return self.results

    def get_results(self, dataset_name=None, model_name=None):

        if self.results is None: self.load_results()

        data = self.results.copy()
        # filter base on dataset
        if dataset_name is not None: data = data[data["dataset_name"] == dataset_name]
        # filter base on model 
        if model_name is not None: data = data[data["model_name"] == model_name]

        return data