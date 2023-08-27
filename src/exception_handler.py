import sys
from src.log_handler import logging


def get_error_message(error, error_detail:sys):

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename 
    error_line = exc_tb.tb_lineno
    error_message = f"Error occured at [{file_name}] at line number [{error_line}]  error message: [{error}] "
    
    return error_message


class CustomException(Exception):

    def __init__(self,error,error_detail):
        super().__init__(error)
        self.error_message = get_error_message(error,error_detail)
        #print("CUSTOM EXCEPTION ACTIVATED")

    def __str__(self):
        return self.error_message







if __name__ == "__main__":
    try:
        print(1/0) 

    except Exception as e :
        new_exception = CustomException(e,sys)
        logging.error(new_exception)
        raise new_exception