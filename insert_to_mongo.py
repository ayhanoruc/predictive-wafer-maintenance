from pymongo import  MongoClient 
from gridfs import GridFS
import csv
import pandas as pd 
import os 

""" # USE CLASSIC APPROACH with insert_many and pandas
def insert_data_classic():
    for file_path in csv_path_list:

        df = pd.read_csv(file_path)
        
        df["source_file"] = os.path.basename(file_path)
        #print(df["source_file"])
        documents= df.to_dict(orient="records")
        collection.insert_many(documents)



def ingest_data_classic():
    for csv_path in csv_path_list:
        file_name = os.path.basename(csv_path)
        query = {"source_file" : file_name}
        result = collection.find(query)
        df= pd.DataFrame(list(result))
        file_path = os.path.join(ingested_data_dir,file_name)
        df.to_csv(file_path,index=False)
        print("INGESTED: {}".format(file_name))
 """


def mongo_connect():
    pass 



# USE FS 
def insert_with_fs(csv_path_list):
    for file_path in csv_path_list:

        with open(file_path,'rb') as f:

            fs_id = fs.put(f,filename=file_path)
            print("inserted file: {} with ID:{}".format(file_path,fs_id))


            
        
def ingest_with_fs(target_dir):
    ingested_data_dir = os.path.join(os.getcwd(),target_dir,"testing")
    os.makedirs(ingested_data_dir,exist_ok=True)
    files= fs.find()
    
    for file in files:
        file_data= fs.get(file._id).read()
        file_name = os.path.basename(file.filename)
        file_path = os.path.join(ingested_data_dir,file_name)

        with open(file_path,"wb") as f:
            f.write(file_data)
            print(f"Saved file {file_name}")
        




if __name__ == "__main__": 

    MONGO_URL = ""
    client = MongoClient('mongodb://localhost:27017/')
    database = client["local"]
    collection = database["wafer-fault-collection"]
    fs = GridFS(database)

    dataset_dir = os.path.join(os.getcwd(),"dataset","prediction-dataset")
    csv_path_list = [os.path.join(dataset_dir,csv) for csv in os.listdir(dataset_dir)]

    insert_with_fs(csv_path_list)

    target_dir = "ingested_data_collection"
    ingest_with_fs(target_dir)

    client.close()