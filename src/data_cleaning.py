import logging
from abc import ABC ,abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    Abstract class defining strategy for data handling
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing Data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data

        Args:
            data (pd.DataFrame): Input DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        try:
            # Drop unnecessary columns
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ], axis=1)

            # Fill missing values for numerical columns with median
            numerical_columns = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
            for column in numerical_columns:
                data[column].fillna(data[column].median(), inplace=True)

            # Fill missing values for 'review_comment_message' with "No review"
            data["review_comment_message"].fillna("No review", inplace=True)

            # Select only numerical columns
            data = data.select_dtypes(include=[np.number])

            # Drop additional unnecessary columns
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data

        except Exception as e:
            logging.error("Error in processing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for Dividing Data into train and Test

    """
    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        X=data.drop(["review_score"],axis=1)
        y=data["review_score"]
        X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test


class DataCleaning:
    """"
    class for preprocessing and dividing into traain and test
    """
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy


    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        """
        handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
        

