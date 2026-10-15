from .BaseDataset import BaseDataset


class GenderDataset(BaseDataset):

    required_columns = BaseDataset.required_columns + [ "masc","fem"]

    @property
    def masculine_translations(self):
        return self.dataframe["masc"].tolist()

    @property
    def feminine_translations(self):
        return self.dataframe["fem"].tolist()