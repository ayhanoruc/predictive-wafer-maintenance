import os,sys 
from src.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

from src.utility.generic import save_object


class ReadyModel:
    
    """
    a trained machine learning model that includes
    both a preprocessor and a model for making predictions.
    """

    def __init__(self,preprocessor, model):

                
        self.preprocessor = preprocessor
        self.model = model 

    
    def predict(self,X,is_testing=True,threshold=0.26):

        X_transformed = self.preprocessor.transform(X, is_testing=is_testing)
        y_pred_proba = self.model.predict_proba(X_transformed)
        y_test_pred = (y_pred_proba[:,1]>threshold).astype(int)
        
        return y_test_pred
    




class ModelResolver:
    def __init__(self, saved_models_dir=SAVED_MODEL_DIR):
        
        self.saved_models_dir = saved_models_dir


    
    def get_best_model_path(self,)->str :
        """
        This method returns the path to the best (latest) model file.
        It first lists all directories (timestamps) within the saved_models_dir.
        It converts these timestamps (directory names) to integers.
        It finds the latest timestamp (which represents the best model) by taking 
        the maximum value from the list of integers.
        It then constructs and returns the path to the best model file by combining the 
        saved_models_dir, the latest timestamp directory, and a constant MODEL_FILE_NAME.
        """
        
        all_parent_dir_names = os.listdir(self.saved_models_dir) # which are just timestamp strings
        timestamps = list(map(int,all_parent_dir_names)) # map them to integer format and put them into a list
        latest_timestamp = max(timestamps) # which will be our best model <- THE LOGIC
        
        return os.path.join(self.saved_models_dir, f"{latest_timestamp}",MODEL_FILE_NAME) # saved_models/84449204/model.pkl
    

    


    def is_a_model_exists(self)->bool:
        """
        This method checks if a model exists in the specified directory.
        It first checks if the saved_models_dir itself exists. If not, it returns False.
        Then, it checks if there are any directories (timestamps) within the saved_models_dir.
        If there are no directories, it returns False.
        Next, it calls the get_best_model_path method to get the path to the best model.
        If the best model file does not exist (based on the path), it returns False.
        Otherwise, it returns True.

        """
        #os.makedirs(os.path.join(self.saved_models_dir),exist_ok=True)
        if not os.path.join(self.saved_models_dir):
            return False
        
        if len(os.listdir(self.saved_models_dir)) == 0:
            return False 
        
        best_model_path = self.get_best_model_path() # refers to "any file"

        if not os.path.exists(best_model_path):
            return False 
        

        return True # otherwise



