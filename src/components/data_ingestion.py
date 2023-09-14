import pandas as pd 

from src.exception_handler import handle_exceptions
from src.log_handler import AppLogger 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact 
from src.data_access.mongo_db import MongoConnect
from src.constants.mongo import MONGO_URL,DB_NAME



class DataIngestionComponent:
        """
    A class for ingesting raw data into the feature store.

    Args:
        data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.

    Attributes:
        log_writer (AppLogger): Logger object for logging data ingestion progress.
        data_ingestion_config (DataIngestionConfig): Data ingestion configuration.

    Methods:
        export_raw_data_into_feature_store: Ingests raw data into the feature store.
        run_data_ingestion: Runs the data ingestion process and returns the DataIngestionArtifact.
    """

        def __init__(self,data_ingestion_config:DataIngestionConfig):
            self.log_writer = AppLogger(phase_name="DataIngestion")

            self.data_ingestion_config = data_ingestion_config
       

       
        @handle_exceptions
        def export_raw_data_into_feature_store(self):
                
            """
            Ingests raw data into the feature store.
            """
            connection = MongoConnect() 
            connection.mongo_connection(MONGO_URL,DB_NAME) 
            #artifact > timestamp > feature_store | ingested_data 
            self.log_writer.handle_logging("RAW DATA INGESTION STARTED")
            connection.ingest_with_fs(target_dir=self.data_ingestion_config.feature_store_dir,collection_name=self.data_ingestion_config.training_collection_name)
            self.log_writer.handle_logging("RAW DATA INGESTION ENDED")
            
        
        """
        ---NOT USED, LEAVED HERE FOR REFERENCE FOR LATER PROJECT ---

        @handle_exceptions
        def split_train_test(self,df:pd.DataFrame)->None:
                
            
                This method takes a DataFrame, splits it into train and test sets using the specified split ratio,
                and then saves the resulting train and test sets to the corresponding file paths provided in
                the data_ingestion_config. If an error occurs during the process, a CustomException is raised,
                and the error is logged.
            
            #takes dataframe , splits it into train and test files then sends them to ingested folder


            train_set, test_set = train_test_split(df,test_size=self.data_ingestion_config.split_ratio,random_state=19)
            self.log_writer.handle_logging("dataset split into train & test")
            write_to_csv(train_set,file_path=self.data_ingestion_config.training_file_path)
            write_to_csv(test_set,file_path=self.data_ingestion_config.testing_file_path)"""


        @handle_exceptions
        def run_data_ingestion(self)->DataIngestionArtifact:
            """
            Runs the data ingestion process.

            Returns:
                DataIngestionArtifact: The data ingestion artifact containing feature store and file paths.
            """
            self.log_writer.handle_logging("-------------ENTERED RAW DATA INGESTION STAGE------------")
            self.export_raw_data_into_feature_store() 

            data_ingestion_artifact= DataIngestionArtifact(
                feature_store= self.data_ingestion_config.feature_store_dir,
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path,

            )
            self.log_writer.handle_logging("CREATED data ingestion artifact successully")

            return data_ingestion_artifact 






