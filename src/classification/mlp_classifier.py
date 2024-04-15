from sklearn.neural_network import MLPClassifier as MLP

from src.classification.classify import Classifier


class MLPClassifier(Classifier):

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

    def load_model(self):
        self.model = MLP(**self.cfg['default'])
