from sklearn.tree import DecisionTreeClassifier

from src.classification.classify import Classifier


class DecisionTree(Classifier):

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

    def load_model(self):
        self.model = DecisionTreeClassifier(**self.cfg['default'])
