import os
import logging
import joblib
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def check_files(out_files, path):
	check_files = []
	for file in os.listdir(path):
		if os.path.isfile(os.path.join(path, file)):
			check_files.append(file)
	for pic in check_files:
		assert not set(check_files).isdisjoint(out_files) == True


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")

	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

	return df

def test_eda(perform_eda, df):
	'''
	test perform eda function
	'''

	try:
		
		perform_eda(df)
		logging.info("Testing perform eda: SUCCESS")
	except Exception as err:
		logging.error("Check the perform_data: FAILURE")
		raise err

	try:
		path = r"./images/eda/"
		list_eda_files = ['Churn_distribution.png', 'Customer_Age_distribution.png', 'heatmap.png', 'Marital_Status_distribution.png', 'total_transaction_distribution.png']
		check_files(list_eda_files, path)
		# check_files = []
		# for file in os.listdir(path):
		# 	if os.path.isfile(os.path.join(path, file)):
		# 		check_files.append(file)
		# for pic in check_files:
		# 	assert not set(check_files).isdisjoint(list_eda_files) == True
		logging.info("Files in the {path}  path validated: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Files not matching ...")
		raise err
	



def test_encoder_helper(encoder_helper, df):
	'''
	test encoder helper
	'''
	cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
					'Income_Category', 'Card_Category']
	try:
		assert isinstance(df, pd.DataFrame)
		assert len(cat_columns) > 0
		df_enc = encoder_helper(df, cat_columns, df['Churn'])
		assert len(df_enc.columns) < len(df.columns)
		logging.info("Encoder check: SUCCESS")
	except AssertionError as err:
		logging.error("Number of columns not matching ...")
		raise err




def test_perform_feature_engineering(perform_feature_engineering, df):
	'''
	test perform_feature_engineering
	'''
	try:
		X_train, X_test, y_train, y_test = perform_feature_engineering(df)
		assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
		assert y_train.shape[0] + y_test.shape[0] == df.shape[0]
		logging.info("Splitting the data into train and test: SUCCESS")
	except AssertionError as err:
		logging.error("Splitting not done properly")
		raise err
		




def test_train_models(train_models, df):
	'''
	test train_models
	'''
	try:
		perform_eda(df)
		cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
						'Income_Category', 'Card_Category']
		df = encoder_helper(df, cat_columns, df['Churn'])
		X_train, X_test, y_train, y_test = perform_feature_engineering(df)
		y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(X_train, X_test, y_train, y_test)
		path_images = r'./images/results/'
		list_results_files = ['feature_importances.png', 'logistic_results.png', 'rf_results.png', 'roc_curve_result.png']
		check_files(list_results_files, path_images)
		path_models = f'./models/'
		list_models = ['logistic_model.pkl','rfc_model.pkl']
		check_files(list_models, path_models)
		logging.info(f"Files in the {path_models} path validated: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Files not matching ...")
		raise err
if __name__ == "__main__":
	df = test_import(import_data)
	test_eda(perform_eda, df)
	test_encoder_helper(encoder_helper, df)
	test_perform_feature_engineering(perform_feature_engineering, df)
	test_train_models(train_models, df)









