# for each new category of evaluation 
# we need to create a class dataset correxponding to the 
# evaluation we want to perform 
# the dataset must be an parquet file  with specific columns 

import pandas as pd


class BaseDataset:

    # Columns required for every QE dataset
    required_columns = ["source", "lang_pair"]

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.validate()

    def validate(self):
        missing = [
            column
            for column in self.required_columns
            if column not in self.dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    @property
    def sources(self):
        return self.dataframe["source"].tolist()

    @property
    def language_pairs(self):
        return self.dataframe["lang_pair"].tolist()

    @classmethod
    def from_parquet(cls, file_path):
        dataframe = pd.read_parquet(file_path)
        return cls(dataframe)