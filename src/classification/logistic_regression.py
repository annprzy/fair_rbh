from sklearn.linear_model import LogisticRegression

from src.classification.classify import Classifier


class LogisticRegressor(Classifier):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

    def load_model(self):
        self.model = LogisticRegression(**self.cfg['default'])