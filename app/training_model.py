from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class TrainingModel(ABC):
    
    @abstractmethod
    def train(self, train_data: list[list[any]], target: list[any]) -> any:
        ...

class LogisticRegressionTrainingModel(TrainingModel):

    def train(self, train_data: list[list[any]], target: list[any]) -> any:
        # Create model
        model = LogisticRegression(solver='liblinear')

        # Train model
        model.fit(train_data, target)
        return model