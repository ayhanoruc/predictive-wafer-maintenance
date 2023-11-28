# PREDICTIVE WAFER MAINTENANCE




# PROBLEM DEFINITION
  
- In semiconductor manufacturing, the production of integrated circuits involves multiple complex processes, including wafer fabrication. Wafers are thin, round substrates made of semiconductor materials, and they undergo various manufacturing steps to create microchips. The quality of these wafers is critical for ensuring the reliability and performance of the final IC's.

- The manufacturing process of semiconducter wafers is susceptible to various defects and faults that can compromise the quality and yield of IC's. These faults can result from contamination, equipment malfunctions or process variations. Detecting and classifying these faults early in the manufacturing process is essential to minimze waste and ensure product quality.

## GOAL
- The goal of this experiment notebook is to findout a succesful / generalized predictive classifier model that can accurately classify semiconducter wafers as either "Good" or "Bad" based on sensor data collected during the manufacturing process. 

## Cost Function
- To address the business priorities and potential consequences of false predictions, a cost function is calculated as follows:

    Cost = 10 * FN (False Negative) + 1 * FP (False Positive)

    This cost function reflects the importance of minimizing false negatives (missed faulty wafers) while considering the cost associated with false positives.


## Technology Stack:

- Python: The primary programming language used for developing the project.

- Flask (2.2.5): to create a web-based user interface.

- Flask Bootstrap (3.3.7.1): a popular front-end framework, to enhance the appearance and functionality of your web application.

- Imbalanced-learn (0.11.0): A library for addressing class imbalance in machine learning datasets.

- NumPy (1.23.5): for numerical operations and working with arrays.

- Optuna (3.3.0): A library for hyperparameter optimization.

- Pandas (2.0.3): for handling and processing datasets.

- PyMongo (4.5.0): A Python driver for MongoDB, a NoSQL database.

- Scikit-learn (1.3.0): for data analysis and modeling, including classification, regression, clustering, and more.

- Setuptools (68.0.0): for packaging Python projects and distributing them.

- XGBoost (1.7.6): An optimized gradient boosting library that is commonly used for supervised learning tasks like classification and regression.

## Infrastructure:

- Flask Web Server: The Flask web application runs on a web server. 

- MongoDB Database: PyMongo is used to interact with a MongoDB database.

- Jupyter Notebook: for data exploration, analysis, and model development in Python.

- Git and GitHub: Git for version control, and GitHub for hosting and collaborating on the project with others.


## Running the Project

To run the project, follow these steps:

1. Clone the repository to your local machine:

    `git clone https://github.com/your-username/predictive-wafer-maintenance.git`


2. Navigate to the project directory:

    `cd predictive-wafer-maintenance`


3. Install the required dependencies using `pip`:

    `pip install -r requirements.txt`


4. Start the Flask web application:

    `python app.py`


5. Access the web application by opening your web browser and navigating to `http://localhost:5000`.

You can now interact with the predictive wafer maintenance system through the web interface.




## Contributing
If you'd like to contribute to this project, please follow these guidelines:

    * Fork the repository.

    * Create a new branch for your feature or bug fix.

    * Make your changes and commit them.

    * Submit a pull request with a clear description of your changes.

### run docker build
- first start Docker engine
- `docker build -t predictive_maintenance_flask_app .`
- since the app depends on mongoDB, go start mongodb compass in local(no need if connecting to mongo remote.)
- `docker run -d -p 5000:5000 predictive_maintenance_flask_app `
