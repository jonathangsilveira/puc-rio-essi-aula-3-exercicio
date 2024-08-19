from abc import ABC, abstractmethod

class AccuracyEvaluation(ABC):

    @abstractmethod
    def evaluate(self, model: any, test_data: list[list[any]], test_target: list[any]) -> float:
        ...

class AccuracyEvaluator(AccuracyEvaluation):

    def evaluate(self, model: any, test_data: list[list], test_target: list[any]) -> float:
        return model.score(test_data, test_target)