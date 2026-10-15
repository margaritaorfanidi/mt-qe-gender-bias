class BaseStatistics:

    @staticmethod
    def compute_scores(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def summarize(score_data):
        raise NotImplementedError
