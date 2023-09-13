
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score , roc_auc_score 

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger
from src.utility.metrics.classification_metrics import cost_function, get_classification_metrics
from src.utility.generic import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ClassificationMetricsArtifact, ModelTrainerArtifact, DataTransformationArtifact
from src.utility.model.model_operations import ReadyModel

import os,sys 

class ModelTrainerComponent:

    def __init__(self,model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):

        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

        self.log_writer = AppLogger("ModelTrainer")

        self.best_model_params = {
            "scale_pos_weight":30,
            "reg_alpha": 0.1
        }
        self.best_threshold = 0.26 # should be centeralized
        #self.model = XGBClassifier(**self.best_model_params)
        self.model = XGBClassifier(scale_pos_weight=30, reg_alpha=0.1)


    def objective(self, trial):
        # Define the hyperparameters to optimize
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600,step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.05),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 30, 100, step=5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0, step = 0.1),
        }

        # Create and train the XGBoost model
        model = XGBClassifier(**params)
        model.fit(self.X_train,self.y_train)

        # Make predictions on the validation set
        y_pred_proba = model.predict_proba(self.X_test)
    
        threshold = trial.suggest_float("threshold", 0.005, 0.4, step=0.005) # Add threshold as a hyperparameter
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)

        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(self.y_test, y_pred)

        cost = cost_function(self.y_test,y_pred)

        # Return the cost as Optuna will try to minimze the objective
        return cost


    
    @handle_exceptions
    def perform_hyperparams_opt(self,):

        self.log_writer.handle_logging("Hyperparams optimization initialized! Cost is being minimized!")

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=5) # will take approximately 4-5 mins depending on your pc config.
        #test etmek için n_trials'ı 5 yap

        # Get the best hyperparameters
        best_params = study.best_params
        self.best_threshold = best_params["threshold"]
        best_params.pop("threshold",None)
        self.best_model_params = best_params
        self.log_writer.handle_logging("The Hyperparams optimization study completed succesfully!")


         

    @handle_exceptions
    def train_model(self,X_train,y_train)->XGBClassifier():

        self.model.fit(X_train,y_train)
        self.log_writer.handle_logging("The model fitted to training data succesfully.")
        return self.model 


    @handle_exceptions    
    def eval_model(self)->tuple: 
        """
        This will return cost related to a specific training/testing stage
        by model object using best_model params and threshold value if exists
        Afterwards, if retraining needed, we should re-assess params and threshold values, for other, satisfactory cases
        we dont need hyperparams tuning
        """

        model_obj = self.train_model(self.X_train,self.y_train)

        y_pred_test_proba = model_obj.predict_proba(self.X_test)

        y_test_pred = (y_pred_test_proba[:,1]>self.best_threshold).astype(int)

        
        f1_score,roc_auc_score,test_cost=  get_classification_metrics(self.y_test,y_test_pred)


        self.log_writer.handle_logging("MODEL Trained/Tested Succesfully.")
        self.log_writer.handle_logging("TESTING : F1-score: {f1_score}, ROC AUC: {roc_auc_score}, Cost: {test_cost} ")
    
        return (f1_score,roc_auc_score,test_cost)

    
    def run_model_trainer(self,)-> ModelTrainerArtifact:
        self.log_writer.handle_logging("-------------ENTERED MODEL TRAINER STAGE------------")
        
        self.log_writer.handle_logging("Reading Training & testing datasets ")
        train_data_file_path= self.data_transformation_artifact.transformed_train_file_path
        test_data_file_path = self.data_transformation_artifact.transformed_test_file_path 

        train_arr = load_numpy_array_data(train_data_file_path)
        test_arr = load_numpy_array_data(test_data_file_path)
        

        self.X_train, self.y_train, self.X_test, self.y_test = (
            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:,:-1],
            test_arr[:,-1]
        )
        print(self.X_train.shape, self.y_train.shape,"testdata:->" ,self.X_test.shape, self.y_test.shape)
        self.log_writer.handle_logging("Training & testing numpy arrays are loaded succesfully!")
        

        f1_score,roc_auc_score,test_cost = self.eval_model()
        
        print(f"test_cost:{test_cost}")

        print(f"roc_auc_score:{roc_auc_score} vs expected_roc_auc_score:{self.model_trainer_config.expected_roc_auc_score}")
        print(f"f1_score:{f1_score} vs expected_f1_score:{self.model_trainer_config.expected_f1_score}")

        if f1_score < self.model_trainer_config.expected_f1_score:
            self.log_writer.handle_logging(f"f1_score:{f1_score} vs expected_f1_score:{self.model_trainer_config.expected_f1_score}")
            self.log_writer.handle_logging("expected cost_score is not satisfied")
            raise Exception("expected cost_score is not satisfied, try to do more Experimentation")


        
        if roc_auc_score < self.model_trainer_config.expected_roc_auc_score:
            self.log_writer.handle_logging(f"roc_auc_score:{roc_auc_score} vs expected_roc_auc_score:{self.model_trainer_config.expected_roc_auc_score}")
            self.log_writer.handle_logging("expected roc_auc_score is not satisfied")
            raise Exception("expected roc_auc_score is not satisfied, try to do more Experimentation")

        self.log_writer.handle_logging("Calculated related metrics succesfully! Model is satistactory and is being saved!")
        # if no exceptions raised -> save the model
        model_file_path = self.model_trainer_config.model_trainer_trained_model_file_path
        model_dir = os.path.dirname(self.model_trainer_config.model_trainer_trained_model_file_path)
        os.makedirs(model_dir, exist_ok=True)
        preprocessor_obj = load_object(self.data_transformation_artifact.preprocessor_object_file_path)

        ready_model = ReadyModel(preprocessor=preprocessor_obj, model = self.model)
        save_object(model_file_path, ready_model)
        self.log_writer.handle_logging(f"ready to predict model is generated & saved succesfully!!! send to {model_file_path}")

        classification_metrics_artifact = ClassificationMetricsArtifact(

            f1_score = f1_score,
            roc_auc_score = roc_auc_score,
            cost_score = test_cost)


        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path = model_file_path,
            train_metric_artifact = None,
            test_metric_artifact = classification_metrics_artifact)
        
        self.log_writer.handle_logging("Model Trainer Artifact Generated Succesfully!")

        return  model_trainer_artifact
        
