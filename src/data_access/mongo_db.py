from pymongo import  MongoClient 
from gridfs import GridFS
import os 
from src.exception_handler import CustomException
from src.log_handler import AppLogger
import logging
from config import PROJECT_ROOT
import pandas as pd 




class MongoConnect:


    def __init__(self,):
        self.log_writer = AppLogger("MongodbConnection")
        


        

    def mongo_connection(self,mongo_url:str,db_name:str)->None:
        try:

            self.client = MongoClient(mongo_url)        
            self.database = self.client[db_name]
            self.gridfs = GridFS(self.database)
            self.log_writer.handle_logging("CONNECTION: SUCCESSFULL")
            
        except Exception as e:
            exception = CustomException(e,sys) 
            self.log_writer.handle_logging(exception,level=logging.ERROR)
            raise exception

    # USE FS 
    def insert_with_fs(self,csv_path_list, collection_name):
        for file_path in csv_path_list:
            metadata = {"collection_name": collection_name}
            with open(file_path,'rb') as f:
                
                file_name = f"{collection_name}_{os.path.basename(file_path)}"
                fs_id = self.gridfs.put(f, filename= file_name,metadata = metadata )
                print("inserted file: {} with ID:{}".format(file_path,fs_id))


                
            
    def ingest_with_fs(self,target_dir,collection_name):
        ingested_data_dir = os.path.join(target_dir)
        os.makedirs(ingested_data_dir,exist_ok=True)
        print("ENTER INGESTION AREA")
        for file in self.gridfs.find({"metadata.collection_name": collection_name}):
            file_data= file.read()
            file_name = os.path.basename(file.filename)
            file_path = os.path.join(ingested_data_dir,file_name)

            with open(file_path,"wb") as f:
                f.write(file_data)
                print(f"Saved file {file_name} from {collection_name}")





""" 

    def ingest_collection(self,COLLECTION_NAME):

        try:

            self.collection = COLLECTION_NAME


        except Exception as e:
            exception = CustomException(e,sys) 
            self.log_writer.handle_logging(exception,level=logging.ERROR)
            raise exception       


 """





if __name__ == "__main__":
    mongo_connect = MongoConnect()
    mongo_connect.mongo_connection(MONGO_URL,DB_NAME)

    #dataset_dir = os.path.join(from_root(),"dataset","prediction-dataset")
    csv_path_list = [os.path.join(TRAINING_RAW_DATASET_DIR,csv) for csv in os.listdir(TRAINING_RAW_DATASET_DIR)]
    target_dir = INGESTED_TRAINING_RAW_DATASET_DIR

    mongo_connect.insert_with_fs(csv_path_list,COLLECTION_NAME)


    mongo_connect.ingest_with_fs(target_dir,COLLECTION_NAME)

    mongo_connect.client.close()