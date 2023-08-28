
import pandas as pd 

import os

from src.exception_handler import CustomException
from src.log_handler import AppLogger


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

