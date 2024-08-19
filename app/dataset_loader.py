
from abc import ABC, abstractmethod

from pandas import DataFrame, read_csv


class DatasetLoader(ABC):

    @abstractmethod
    def load(self, source: any, attributes: list[str]) -> DataFrame:
        ...

class CSVDatasetLoader(DatasetLoader):

    def load(self, source: any, attributes: list[str]) -> DataFrame:
        return read_csv(str(source), skiprows=0, delimiter=',')