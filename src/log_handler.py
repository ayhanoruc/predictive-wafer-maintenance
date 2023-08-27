import logging 
import os 
from datetime import datetime 



current_time = datetime.now()
LOG_FILE_NAME = f"{current_time.strftime('%d-%m-%Y-%H-%M')}.log"


logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)
log_file_path = os.path.join(logs_path,LOG_FILE_NAME)


logging.basicConfig(

    filename=log_file_path,
    format="[%(asctime)s - %(name)s -- %(levelname)s -- %(message)s]",
    level = logging.INFO
)


