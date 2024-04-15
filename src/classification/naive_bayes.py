from sklearn.naive_bayes import GaussianNB

from src.classification.classify import Classifier


class NaiveBayes(Classifier):

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

    def load_model(self):
        self.model = GaussianNB()
