import logging
import pandas as pd
from zenml import step



@step
def evaluate_model(df:pd.DataFrame)->None:
    """
    evaluatted the model on ingested data .

    Args:
    df: the ingested data
    
    """
    pass