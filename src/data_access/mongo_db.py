from pymongo import  MongoClient 
from gridfs import GridFS
import os 
from src.exception_handler import handle_exceptions
from src.log_handler import AppLogger



class MongoConnect:
    """
    This class provides methods for connecting to a MongoDB database, inserting data into the database using GridFS, and ingesting data from GridFS.

    Attributes:
        log_writer (AppLogger): An instance of the AppLogger class for logging.
    """


    def __init__(self,):
        self.log_writer = AppLogger("MongodbConnection")
        


        
    @handle_exceptions
    def mongo_connection(self,mongo_url:str,db_name:str)->None:
        """
        Establish a connection to a MongoDB database.

        Args:
            mongo_url (str): The MongoDB connection URL.
            db_name (str): The name of the database to connect to.

        Sets:
            client (MongoClient): The MongoClient instance for the connection.
            database (MongoDatabase): The MongoDB database instance.
            gridfs (GridFS): The GridFS instance for the database.
        """

        self.client = MongoClient(mongo_url)        
        self.database = self.client[db_name]
        self.gridfs = GridFS(self.database)
        self.log_writer.handle_logging(f"MONGO CONNECTION: connected to {mongo_url},{db_name} succesfully!")
            

    # USE FS 
    @handle_exceptions
    def insert_with_fs(self,csv_path_list, collection_name):
        """
        Insert CSV files into MongoDB using GridFS.

        Args:
            csv_path_list (list): A list of file paths to the CSV files to insert.
            collection_name (str): The name of the MongoDB collection to store the files in.

        Inserts CSV files into MongoDB using GridFS, associating them with a collection name.
        """
        metadata = {"collection_name": collection_name} #Metadata containing the collection name is associated with each file.
        for file_path in csv_path_list:
            
            with open(file_path,'rb') as f:
                
                file_name = f"{collection_name}_{os.path.basename(file_path)}"
                fs_id = self.gridfs.put(f, filename= file_name,metadata = metadata )
                self.log_writer.handle_logging("inserted file: {} with ID:{}".format(file_path,fs_id))
                print("inserted file: {} with ID:{}".format(file_path,fs_id))


                
    @handle_exceptions 
    def ingest_with_fs(self,target_dir,collection_name):
        """
        Ingest files from MongoDB GridFS into a target directory.

        Args:
            target_dir (str): The directory to which the ingested files will be saved.
            collection_name (str): The name of the MongoDB collection from which to ingest files.
        """
        ingested_data_dir = os.path.join(target_dir) #review this line
        os.makedirs(ingested_data_dir,exist_ok=True)
        for file in self.gridfs.find({"metadata.collection_name": collection_name}):
            file_data= file.read()
            file_name = os.path.basename(file.filename)
            file_path = os.path.join(ingested_data_dir,file_name)

            with open(file_path,"wb") as f:
                f.write(file_data)
                self.log_writer.handle_logging(f"Saved file {file_name} from {collection_name} to {ingested_data_dir}")



if __name__ == "__main__":
    pass 