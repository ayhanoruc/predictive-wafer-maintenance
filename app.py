from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import os,sys 
import datetime 
from config import PROJECT_ROOT


app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = 'uploads_dir'
app.config['STATIC_FOLDER'] = 'static'  

@app.route('/', methods=['GET','POST'])
def index():    
    return render_template('index.html')



@app.route('/training', methods=['GET', 'POST'])
def training():
    upload_status = None  # Initialize upload status

    if request.method == 'POST':
        uploades_dir = os.path.join(PROJECT_ROOT, "uploads_dir")
        os.makedirs(uploades_dir, exist_ok=True)
        if 'files[]' not in request.files:
            upload_status = "No files were uploaded."
        else:
            uploaded_files = request.files.getlist('files[]')
            for file in uploaded_files:
                file_path = os.path.join(uploades_dir, file.filename)
                file.save(file_path)
            upload_status = "Files uploaded successfully!"

    return render_template('training.html', upload_status=upload_status)



@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    prediction_status = None  # Initialize upload status

    if request.method == 'POST':
        uploades_dir = os.path.join(PROJECT_ROOT, "uploads_dir")
        os.makedirs(uploades_dir, exist_ok=True)
        
        if 'files[]' not in request.files:
            prediction_status = "No files were uploaded."
            
        else:
                
            uploaded_files = request.files.getlist('files[]')

            for file in uploaded_files:
                file_path = os.path.join(uploades_dir, file.filename)
                file.save(file_path)
        
            prediction_status = "Prediction successful!"
        

    return render_template('prediction.html', prediction_status=prediction_status)




if __name__ == '__main__':
    app.run(debug=True)
