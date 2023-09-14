from flask import Flask, render_template, url_for, request, jsonify
from flask_bootstrap import Bootstrap
import os,sys 
import datetime 
from config import PROJECT_ROOT
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline


app = Flask(__name__)
Bootstrap(app)
#app.config['UPLOAD_FOLDER'] = 'prediction_dataset_dir'
app.config['STATIC_FOLDER'] = 'static'  

@app.route('/', methods=['GET','POST'])
def index():    
    return render_template('index.html')



@app.route('/training', methods=['GET', 'POST'])
def training():
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




@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

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



def get_prediction_results():

    #y_pred,best_model_metrics = PredictionPipeline().start_prediction_pipeline()
    print("prediction started")
    prediction_result = PredictionPipeline().start_prediction_pipeline()[0].tolist()
    #print(prediction_result)

    return prediction_result


@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    
    prediction_result = get_prediction_results()
    #print(prediction_result)

    response ={

        'prediction_result': prediction_result
        
        }
  

    return jsonify(response)


def train_model():
    print("model training started")
    training_pipeline = TrainingPipeline()
    model_eval_dict = training_pipeline.run_training_pipeline(is_manual_ingestion=True)

    return model_eval_dict

@app.route('/manual_training', methods=['POST'])
def manual_training():
    
    model_eval_dict = train_model()

    response = {'prediction_result': model_eval_dict}

    return jsonify(response)





if __name__ == '__main__':
    app.run(debug=True)
    



