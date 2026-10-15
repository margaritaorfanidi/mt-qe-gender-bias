from .BaseDataset import BaseDataset



class GeneralDataset(BaseDataset):

    required_columns = BaseDataset.required_columns + ["mt"]

    @property
    def translations(self):
        return self.dataframe["mt"].tolist()

    @property
    def scores(self):
        if "score" not in self.dataframe.columns:
            return None

        return self.dataframe["score"].tolist()
