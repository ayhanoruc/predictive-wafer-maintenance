from flask import Flask, render_template, url_for, request, jsonify
from flask_bootstrap import Bootstrap

from config import PROJECT_ROOT
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.log_handler import AppLogger
from src.exception_handler import handle_exceptions

import os



log_writer = AppLogger("USER INTERACTION")


app = Flask(__name__)
Bootstrap(app)

app.config['STATIC_FOLDER'] = 'static'  

@handle_exceptions
@app.route('/', methods=['GET','POST'])
def index():    
    """
    Main page of the web application.

    Returns:
        render_template: Renders the index.html template.
    """
    return render_template('index.html')


@handle_exceptions
@app.route('/training', methods=['GET', 'POST'])
def training():

    """
    This function handles both GET and POST requests. For GET requests, it renders the 'training.html' template.
    For POST requests, it processes uploaded files, saves them, and provides an upload status.

    Returns:
        render_template: Renders the 'training.html' template with upload status, ROC AUC score, F1 score, and evaluation metrics.
    """
    upload_status = None  # Initialize upload status
    roc_auc_score = None 
    f1_score = None 
    eval_metrics = None 
    if request.method == 'POST':
        uploades_dir = os.path.join(PROJECT_ROOT, 'uploaded_feature_store')
        os.makedirs(uploades_dir, exist_ok=True)
        uploaded_files = request.files.getlist('files[]')
        
        if len(uploaded_files)<=1:
            upload_status = "No files were uploaded."
        else:
            
            for file in uploaded_files:
                file_path = os.path.join(uploades_dir, file.filename)
                file.save(file_path)
            upload_status = "Files uploaded successfully!"

    return render_template('training.html', upload_status=upload_status, roc_auc_score = roc_auc_score, f1_score = f1_score, eval_metrics = eval_metrics)



@handle_exceptions
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """
    This function handles both GET and POST requests. For GET requests, it renders the 'prediction.html' template.
    For POST requests, it processes uploaded files, saves them, and provides an upload status.

    Returns:
        render_template: Renders the 'prediction.html' template with upload status and prediction results.
    """

    upload_status = None  # Initialize upload status
    prediction_results = None 

    if request.method == 'POST':
        uploades_dir = os.path.join(PROJECT_ROOT, 'prediction_dataset_dir')
        os.makedirs(uploades_dir, exist_ok=True)
        
        uploaded_files = request.files.getlist('files[]')

        if len(uploaded_files)<=1:
            upload_status = "No files were uploaded."
        else:
            for file in uploaded_files:
                file_path = os.path.join(uploades_dir, file.filename)
                file.save(file_path)
            upload_status = "Prediction files uploaded successfully!"
            #return render_template('prediction.html', upload_status=upload_status, prediction_results= prediction_results)
            
        """i will start model prediction pipeline that is called from another module(which is ready to use)
           the prediction pipeline will return predictions, then i need to update the right side predictions box after getting
           predictions.
        """


    return render_template('prediction.html', upload_status=upload_status, prediction_results= prediction_results)


@handle_exceptions
def get_prediction_results():

    #y_pred,best_model_metrics = PredictionPipeline().start_prediction_pipeline()
    log_writer.handle_logging("Prediction pipeline started")
    prediction_result = PredictionPipeline().start_prediction_pipeline()[0].tolist()
    log_writer.handle_logging("Prediction pipeline completed")
    #print(prediction_result)

    return prediction_result

@handle_exceptions
@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    """
    Get prediction results using the PredictionPipeline.

    This function initiates the prediction pipeline to obtain predictions and returns the prediction results as a json object.

    Returns:
        json object: A dict of prediction results.
    """
    
    prediction_result = get_prediction_results()

    response ={

        'prediction_result': prediction_result
        
        }
  

    return jsonify(response)


@handle_exceptions
def train_model():
    """
    Train a machine learning model using the TrainingPipeline.
    Returns:
        dict: A dictionary containing model evaluation metrics.
    """
    log_writer.handle_logging("Model training started")
    training_pipeline = TrainingPipeline()
    model_eval_dict = training_pipeline.run_training_pipeline(is_manual_ingestion=True)
    log_writer.handle_logging("Model training completed")
    return model_eval_dict


@handle_exceptions
@app.route('/manual_training', methods=['POST'])
def manual_training():
    """
    This route handles the manual training request from the user interface. It calls the `train_model` function
    to initiate model training with manual data ingestion and then returns the model evaluation metrics in a JSON response.

    Returns:
        jsonify: A JSON response containing the model evaluation metrics.
    """
    
    model_eval_dict = train_model()

    response = {'prediction_result': model_eval_dict}

    return jsonify(response)





if __name__ == '__main__':
    app.run(debug=True)
    



