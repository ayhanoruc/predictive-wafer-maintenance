import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

import os, sys 
import logging



from src.exception_handler import CustomException
from src.log_handler import AppLogger 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact 
from src.constants.training_pipeline import TRAINING_SCHEMA_FILE_PATH
from src.data_access.mongo_db import MongoConnect
from src.constants.mongo import MONGO_URL,DB_NAME,TRAINING_COLLECTION_NAME

from src.utility.generic import write_to_csv, export_csv_to_df,merge_csv_files
#from src.constants.training_pipeline impo


def handle_exceptions(func):
    def wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e :
            exc = CustomException(e,sys)
            args[0].log_writer.handle_logging(exc,level=logging.ERROR)
            raise exc 


class DataIngestionComponent:
    
    
        def __init__(self,data_ingestion_config:DataIngestionConfig):
            self.log_writer = AppLogger(phase_name="DataIngestion")

            self.data_ingestion_config = data_ingestion_config
            #self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)







        #@handle_exceptions
        def export_raw_data_into_feature_store(self):
            connection = MongoConnect() 
            connection.mongo_connection(MONGO_URL,DB_NAME) 
            #artifact > timestamp > feature_store | ingested_data 
            print("connection succesfull")
            connection.ingest_with_fs(target_dir=self.data_ingestion_config.feature_store_dir,collection_name=self.data_ingestion_config.training_collection_name)
            print("ingestion with fs successfull")
            


        

        #@handle_exceptions
        def split_train_test(self,df:pd.DataFrame)->None:
            """
                This method takes a DataFrame, splits it into train and test sets using the specified split ratio,
                and then saves the resulting train and test sets to the corresponding file paths provided in
                the data_ingestion_config. If an error occurs during the process, a CustomException is raised,
                and the error is logged.
            """
            #takes dataframe , splits it into train and test files then sends them to ingested folder

            train_set, test_set = train_test_split(df,test_size=self.data_ingestion_config.split_ratio,random_state=19)
            
            write_to_csv(train_set,file_path=self.data_ingestion_config.training_file_path)
            write_to_csv(test_set,file_path=self.data_ingestion_config.testing_file_path)


        





        #@handle_exceptions
        def run_data_ingestion(self)->DataIngestionArtifact:

            self.export_raw_data_into_feature_store() 
            #here merge operation should come into play: merge multiple dataframes into single dataframe
            #csv_path_list=[] # this should come from validated data artifact
            #merge_csv_files(csv_path_list)
            #self.split_train_test(df)
            print("all files ingested")
            data_ingestion_artifact= DataIngestionArtifact(
                feature_store= self.data_ingestion_config.feature_store_dir,
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path,

            )
            print(data_ingestion_artifact)
            return data_ingestion_artifact 






