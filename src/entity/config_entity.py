from datetime import datetime 
import os 
from src.constants import training_pipeline
from config import PROJECT_ROOT


class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        self.timestamp = timestamp.strftime("%d_%m_%Y_%H_%M")
        self.pipeline_name = training_pipeline.PIPELINE_NAME 
        self.artifact_dir = os.path.join(PROJECT_ROOT,training_pipeline.ARTIFACT_DIR_NAME,self.timestamp)




class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
        self.training_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TRAIN_FILE_NAME) # These are used after split operation
        self.testing_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TEST_FILE_NAME)
        self.split_ratio = training_pipeline.DATA_INGESTION_SPLIT_RATIO
        self.training_collection_name = training_pipeline.TRAINING_COLLECTION_NAME
        self.unseen_collection_name = training_pipeline.UNSEEN_COLLECTION_NAME # we will ingest this after the best model obtained, this will play as real-time unseen data,(monitoring purpose)


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR)
        self.data_validation_training_dir = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_TRAINING_DIR)
        self.data_validation_unseen_dir = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_UNSEEN_DIR)

        self.valid_training_data_dir = os.path.join(self.data_validation_training_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_training_data_dir = os.path.join(self.data_validation_training_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)

        self.valid_unseen_data_dir = os.path.join(self.data_validation_unseen_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_unseen_data_dir = os.path.join(self.data_validation_unseen_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)

        self.mongo_valid_collection_name = training_pipeline.MONGO_VALID_COLLECTION_NAME
        self.mongo_invalid_collection_name = training_pipeline.MONGO_INVALID_COLLECTION_NAME
        

        