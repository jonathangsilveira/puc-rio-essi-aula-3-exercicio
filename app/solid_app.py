
from pandas import DataFrame
from dataset_loader import CSVDatasetLoader

from pre_processor import PreProcessor

from training_model import LogisticRegressionTrainingModel

from accuracy_evaluation import AccuracyEvaluator

url: str = 'https://raw.githubusercontent.com/tatianaesc/datascience/main/diabetes.csv'
columns: list[str] = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

loader = CSVDatasetLoader()
dataset: DataFrame = loader.load(source=url, attributes=columns)

test_size: float = 0.33
seed: int = 7
processor = PreProcessor()
train_data, test_data, train_target, test_target = processor.pre_process(
    dataset=dataset, 
    test_size=test_size, 
    seed=seed
)

model = LogisticRegressionTrainingModel()
trained_model = model.train(train_data=train_data, target=train_target)

evaluator = AccuracyEvaluator()
score = evaluator.evaluate(
    model=trained_model, 
    test_data=test_data, 
    test_target=test_target
)
print(score)

# Post-processing