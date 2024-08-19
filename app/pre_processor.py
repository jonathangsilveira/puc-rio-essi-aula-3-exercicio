from pandas import DataFrame
from sklearn.model_selection import train_test_split

class PreProcessor:

    def pre_process(self, dataset: DataFrame, 
                    test_size: float, seed: int) -> list[any]:
        # Clean up data and remove outliers

        # Feature selection

        # Split up train and test data
        x_train, x_test, y_train, y_test = self.__prepare_holdout__(dataset, test_size, seed)

        # Normalization/Stardarization
        return (x_train, x_test, y_train, y_test)

    def __prepare_holdout__(self, dataset: DataFrame, test_size: float, 
                            seed: int) -> list[any]:
        """
        """
        data: list[list[any]] = dataset.to_numpy()
        predictor: list[list[any]] = data[:,0:-1] # From first column index to the last one
        target: list[any] = data[:,-1] # It gets each cell from the last column index.
        
        return train_test_split(predictor, target, test_size=test_size, random_state=seed)
    
