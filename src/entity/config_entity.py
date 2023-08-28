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
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR_NAME)
        self.training_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,training_pipeline.TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,training_pipeline.TEST_FILE_NAME)
        self.split_ratio = training_pipeline.DATA_INGESTION_SPLIT_RATIO
        self.training_collection_name = training_pipeline.TRAINING_COLLECTION_NAME
        self.unseen_collection_name = training_pipeline.UNSEEN_COLLECTION_NAME # we will ingest this after the best model obtained, this will play as real-time unseen data,(monitoring purpose)

