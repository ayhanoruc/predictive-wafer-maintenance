
import pandas as pd 
import re
import os
import json 
from src.exception_handler import CustomException
from src.log_handler import AppLogger
import dill

def export_csv_to_df(csv_path:str)->pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df 


def merge_csv_files(csv_path_list:str)->pd.DataFrame:

    merged_df = pd.concat(objs=csv_path_list)
    return merged_df

    

def write_to_csv(dataframe:pd.DataFrame,file_path:str)->None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    dataframe.to_csv(file_path)


def read_json_file(file_path:str)->dict:
    with open(file_path,"r") as json_file:
        return json.load(json_file)


def create_regex():
    #wafer-fault-training-collection_...
    filename_pattern = r"^wafer-raw-training-collection_Wafer_(?P<DateStamp>\d{8})_(?P<TimeStamp>\d{6})\.csv$"
    re_object = re.compile(filename_pattern,re.IGNORECASE)
    return re_object

def check_regex_match(re_object,file_name):
    match = re_object.match(file_name)
    return bool(match)


def save_numpy_array_data(file_path:str, array:np.array)->None:
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    dir_path = os.path.dirname(file_path)
    os.make_dirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as f:
        np.save(f,array)
    

def load_numpy_array_data(file_path: str)->np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    with open(file_path,"rb") as f:
        return np.load(f)


def save_object(file_path:str, obj:object)->None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, "wb") as f:
        dill.dump(obj, f)


def load_object(file_path:str)->object:
    # raise exception if path not exists

    with open(file_path, "rb") as f:
        return dill.load(f)

