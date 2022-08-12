# Table of Contents
    1. Introduction - the project's aim
    2. Technologies
    3. Files and data description
    3. Launch



## 1. Introduction

- This Project is called **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity. This project is aimed at showing the skills needed in a project to perform regular ML tasks. This covers ML data reading, data split, model training by tuning hyperparameters,  model evaluation, and model saving. The second part of this project addresses the different tests that need to be performed to conform with the coding standards

## 2. technologies
Here are the technologies used:
    * Visual Studio Code (coding IDE)
    * Github for Version Control
    * Python Packages: 
                        * scikit-learn==0.24.1
                        * shap==0.40.0
                        * joblib==1.0.1
                        * pandas==1.2.4
                        * numpy==1.20.1
                        * matplotlib==3.3.4
                        * seaborn==0.11.2
                        * pylint==2.7.4
                        * autopep8==1.5.6

## 3. Files and data description
Files are described as follows:
* data
    * bank_data.csv
* images
    * eda
        * churn_distribution.png: Gives a distribution of the response variable as a bar chart
        * customer_age_distribution.png: gives a distribution of the customer Age
        * heatmap.png: gives a correlation matrix plot of the variables in the data
        * marital_status_distribution.png: gives a distribution of the marital status
        * total_transaction_distribution.png
    * results
        * feature_importance.png: give a feature importance of the features in the models
        * logistics_results.png: highlights the logistic regression model metrics
        * rf_results.png: highlights the random forest model metrics
        * roc_curve_result.png: roc curve of both models
    * logs
        * churn_library.log: log file of the churn_library file
    * models
        * logistic_model.pkl: logistic model pickle file
        * rfc_model.pkl: random forest model pickle file 
churn_library.py
churn_notebook.ipynb
churn_script_logging_and_tests.py
README.md


## 4. Launch
How do you run your files? What should happen when you run your files?
Running the churn_library.py: $python churn_library.py
Running the churn_script_logging_and_tests.py:$python churn_script_logging_and_tests.py
Testing the churn_script_logging_and_tests.py: 1. Run autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
                                               2. Run pylint: pylint churn_script_logging_and_tests.py



