from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_Data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    # Ingest data
    df = ingest_data(data_path)

    # Clean data
    clean_Data(df)
    

    # Train model
    train_model(df)

    # Evaluate model
    evaluate_model(df)

    
