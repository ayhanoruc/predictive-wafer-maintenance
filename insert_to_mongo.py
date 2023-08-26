from pymongo import  MongoClient 
from gridfs import GridFS
import csv
import pandas as pd 
import os 
import time 

# USE CLASSIC APPROACH with insert_many and pandas
def insert_data_classic():
    for file_path in csv_path_list:

        df = pd.read_csv(file_path)
        
        df["source_file"] = os.path.basename(file_path)
        #print(df["source_file"])
        documents= df.to_dict(orient="records")
        collection.insert_many(documents)

ingested_data_dir_name = "ingested_data_collection"
ingested_data_dir = os.path.join(os.getcwd(),ingested_data_dir_name)
os.makedirs(ingested_data_dir,exist_ok=True)


def ingest_data_classic():
    for csv_path in csv_path_list:
        file_name = os.path.basename(csv_path)
        query = {"source_file" : file_name}
        result = collection.find(query)
        df= pd.DataFrame(list(result))
        file_path = os.path.join(ingested_data_dir,file_name)
        df.to_csv(file_path,index=False)
        print("INGESTED: {}".format(file_name))



# USE FS 
def insert_with_fs():
    for file_path in csv_path_list:

        with open(file_path,'rb') as f:

            fs_id = fs.put(f,filename=file_path)
            print("inserted file: {} with ID:{}".format(file_path,fs_id))


dataset_dir = os.path.join(os.getcwd(),"dataset")
csv_path_list = [os.path.join(dataset_dir,csv) for csv in os.listdir(dataset_dir)]


def insert_with_gridfs():

    for file_path in csv_path_list:
        with open(file_path,"rb") as f:
            fs_id = fs.put(f,file_path)
            

def ingest_with_gridfs(fs):
    ingested_data_dir_name = "ingested_data_collection"
    ingested_data_dir = os.path.join(os.getcwd(),ingested_data_dir_name)
    os.makedirs(ingested_data_dir,exist_ok=True)
    files = fs.find()

    for file in files:
        file_data = fs.get(file._id).read()
        file_name = os.path.basename(file.filename)
        file_path = os.path.join(ingested_data_dir,file_name)
        with open(file_path,"wb") as f:
            f.write(file_data)
            
        



        
def ingest_with_fs():
    ingested_data_dir_name = "ingested_data_collection"
    ingested_data_dir = os.path.join(os.getcwd(),ingested_data_dir_name)
    os.makedirs(ingested_data_dir,exist_ok=True)
    files= fs.find()
    
    for file in files:
        file_data= fs.get(file._id).read()
        file_name = os.path.basename(file.filename)
        file_path = os.path.join(ingested_data_dir,file_name)

        with open(file_path,"wb") as f:
            f.write(file_data)
            print(f"Saved file {file_name}")
        


""" ingested_data_dir_name = "ingested_data_collection"
ingested_data_dir = os.path.join(os.getcwd(),ingested_data_dir_name)
print(ingested_data_dir)
os.makedirs(ingested_data_dir,exist_ok=True)

files = fs.find()


for file in files:
    file_data = fs.get(file._id).read()
    file_name = os.path.basename(file.filename)
    file_path = os.path.join(ingested_data_dir, file_name)  # Construct the complete file path
    
    with open(file_path, "wb") as f:
        f.write(file_data)
        print(f"Saved file: {file_name} to {file_path}")

client.close()
 """




if __name__ == "__main__": 

    MONGO_URL = ""
    client = MongoClient('mongodb://localhost:27017/')
    database = client["local"]
    collection = database["wafer-fault-collection"]
    fs = GridFS(database)
    dataset_dir = "./dataset"
    csv_path_list = [f"./dataset/{csv}" for csv in os.listdir(dataset_dir)]
    t1 = time.time()
    #insert_data_classic()
    insert_with_fs()
    t2 = time.time()
    insert_t = t2-t1 
    t1 = time.time()
    #ingest_data_classic()
    ingest_with_fs()
    t2 = time.time()
    ingest_t = t2-t1
    print(insert_t,ingest_t) 

    client.close()