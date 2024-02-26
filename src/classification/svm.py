from sklearn.svm import SVC

from src.classification.classify import Classifier


class SVM(Classifier):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

    def load_model(self):
        self.model = SVC(**self.cfg['default'])