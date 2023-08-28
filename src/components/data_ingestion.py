import pandas as pd 

import os, sys 
import logging



from src.exception_handler import CustomException
from src.log_handler import AppLogger 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact 
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from src.data_access.mongo_db import MongoConnect


class DataIngestionComponent:
    
    
        def __init__(self,data_ingestion_config:DataIngestionConfig):
            self.log_writer = AppLogger(phase_name="DataIngestion")
            try:
                self.data_ingestion_config = data_ingestion_config
                #self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
                print(1/0)
            
    
            except Exception as e :
                new_exception = CustomException(e,sys)
                self.log_writer.handle_logging(new_exception,level=logging.ERROR)
                raise new_exception
    

        def export_raw_data_into_feature_store(self)->pd.DataFrame:
            pass 


        def split_train_test(self,df:pd.DataFrame)->None:
            pass 


        def run_data_ingestion(self)->DataIngestionArtifact:
            pass 







