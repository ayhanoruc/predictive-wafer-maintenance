import pandas as pd
import numpy as np 

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek 
from sklearn.base import BaseEstimator , TransformerMixin

import os, sys 
import dill 

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.utility.generic import save_numpy_array_data, save_object


valid_train_dataset_dir = "../valid_feature_store/valid_training_data/" # test purpose
valid_predict_dataset_dir = "../valid_feature_store/valid_predict_data/" # test purpose


class TrainingPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,use_y = True):
        self.use_y = use_y

    def fit(self,X,y=None):
        if self.use_y:
            self.y = y 

        return self 
    
    def transform(self,X,is_testing=False):
        X_transformed = DataTransformationComponent.drop_zero_std(X)
        X_transformed = DataTransformationComponent.drop_duplicated_cols(X_transformed)
        X_transformed = X_transformed[important_cols]

        imputer = SimpleImputer(strategy="constant",fill_value=0)
        X_transformed = imputer.fit_transform(X_transformed)

        if not is_testing:
            X_transformed , self.y = DataTransformationComponent.handle_imbalance(X_transformed,self.y)
            r_scaler = RobustScaler()
            X_transformed = r_scaler.fit_transform(X_transformed)
            return X_transformed, self.y 
            
        r_scaler = RobustScaler()
        X_transformed = r_scaler.fit_transform(X_transformed)

        return X_transformed



class DataTransformationComponent:
    def __init__(self, data_transformation_config:DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        self.data_transformation_config  = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        #self.important_cols = DataTransformationComponent.get_important_cols()

        self.log_writer = AppLogger("DataTransformation")
        
    @staticmethod
    def restore_original_data()->pd.DataFrame:
        csv_file_list = os.listdir(valid_train_dataset_dir)
        df_merged = pd.DataFrame()
        for file in csv_file_list:
            file_path = os.path.join(valid_train_dataset_dir,file)
            df = pd.read_csv(file_path)
            df_merged = pd.concat(objs=[df_merged,df],ignore_index=True) # merged around axis=0
            df_merged.drop(columns=["Wafer"],inplace=True)
            filt = df_merged["Good/Bad"]==1
            df_merged["Good/Bad"] = np.where(filt,1,0)

        return df_merged 
        
    @staticmethod
    def drop_zero_std(dataframe:pd.DataFrame)->pd.DataFrame:
        zero_std_cols = dataframe.columns[dataframe.std()==0]
        dataframe2 = dataframe.drop(columns=zero_std_cols)
        return dataframe2


    @staticmethod
    def drop_highly_correlated_columns(dataframe:pd.DataFrame,corr_threshold=0.95,count_threshold=3):

        corr_matrix= dataframe.corr(method="pearson")
        filt = (abs(corr_matrix)>corr_threshold) & (abs(corr_matrix)<1.00)
        corr_counts = filt.sum(axis=1)
        #highly_correlated = corr_counts[cor_counts > count_threshold].sort_values(ascending=False)
        highly_correlated_cols = dataframe.columns[corr_counts>count_threshold]

        return dataframe.drop(columns=highly_correlated_cols)
    
    @staticmethod
    def drop_duplicated_cols(dataframe:pd.DataFrame)->pd.DataFrame:
        duplicated_cols = dataframe.T[dataframe.T.duplicated()].index
        return dataframe.drop(columns=duplicated_cols)

    @staticmethod
    def handle_imbalance(X,y):
        smt = SMOTETomek(random_state=11,sampling_strategy="minority")
        return smt.fit_resample(X,y)

    @staticmethod
    def create_train_test(dataframe):
        X= dataframe.drop("Good/Bad", axis="columns")
        y= dataframe["Good/Bad"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=11,stratify=y)

        return X_train, X_test, y_train, y_test

    @staticmethod 
    def get_important_cols(dataframe:pd.DataFrame)->list:
        df_corr = pd.DataFrame()
        df_corr["Sensor_id"] = dataframe.columns[:-1].tolist()
        corr_score = [abs(round(dataframe[[col,"Good/Bad"]].corr().iloc[0,1],4)) for col in dataframe.columns[:-1]]
        df_corr["corr"] = corr_score
        important_cols = df_corr.sort_values(by="corr",ascending=False)[:200]["Sensor_id"].to_list()
        return important_cols   # without Target Column



    def get_preprocessor(self):

        preprocessor = TrainingPreprocessor()
  
        return preprocessor


    
    def run_data_transformation(self,)->DataTransformationArtifact:

        valid_train_dataset_dir = self.data_validation_artifact.valid_data_dir

        train_df_merged = self.restore_original_data() # use this valid_train_dataset_dir
        # unseen_df = self.restore_original_data("valid_unseen_dataset_dir") # UNSEEN DATASET, may be we can store this outside the feature store
        # bunları prediction pipeline'da özel olarak belirtmek daha makul, çünkü burası direkt training data preprocessingine ait.

        #X_unseen = unseen_df.drop("Good/Bad", axis="columns")
        #y_unseen= unseen_df["Good/Bad"]

        X_train, X_test, y_train, y_test = DataTransformationComponent.create_train_test(train_df_merged)

        important_cols = DataTransformationComponent.get_important_cols(train_df_merged)
        preprocessor = self.get_preprocessor()

        preprocessor_obj= preprocessor.fit(X_train,y_train) 

        X_train_transformed, y_train_transformed = preprocessor_obj.transform(X_train, is_testing=False)
        X_test_transformed = preprocessor_obj.transform(X_test, is_testing=True)


        #save numpy array data
        train_arr = np.c_[np.array(X_train_transformed), np.array(y_train_transformed)]
        test_arr = np.c_[np.array(X_test_transformed),np.array(y_test)]

        save_numpy_array_data(self.data_transformation_config.data_transformation_transformed_train_file_path,
                              array=train_arr)

        save_numpy_array_data(self.data_transformation_config.data_transformation_transformed_test_file_path,
                             array=test_arr)

        #save transformer object
        save_object(self.data_transformation_config.data_transformation_object_file_path,
                    obj= preprocessor_obj )

        #return transformation artifact
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path= self.data_transformation_config.data_transformation_transformed_train_file_path,
            transformed_test_file_path = self.data_transformation_config.data_transformation_transformed_test_file_path,
            preprocessor_object_file_path = self.data_transformation_config.data_transformation_object_file_path
        )
        return data_transformation_artifact



        









