####################################################################
################ Author: Oscar Adimi ###############################
################ Date: 8/8/2022 ####################################
################ Description: Testing Churn Library file ###########

import os
import logging
import seaborn as sns
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models


sns.set()
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def check_files(out_files, path):
    check_files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            check_files.append(file)
    assert not set(check_files).isdisjoint(out_files)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        in_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert in_data.shape[0] > 0
        assert in_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return in_data


def test_eda(perform_eda, in_data_eda):
    '''
    test perform eda function
    '''

    try:

        perform_eda(in_data_eda)
        logging.info("Testing perform eda: SUCCESS")
    except Exception as err:
        logging.error("Check the perform_data: FAILURE")
        raise err

    try:
        path = r"./images/eda/"
        list_eda_files = [
            'Churn_distribution.png',
            'Customer_Age_distribution.png',
            'heatmap.png',
            'Marital_Status_distribution.png',
            'total_transaction_distribution.png']
        check_files(list_eda_files, path)
        logging.info(f"Files in the {path}  path validated: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Files not matching ...")
        raise err


def test_encoder_helper(encoder_helper, in_data_helper):
    '''
    test encoder helper
    '''
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    try:
        assert isinstance(in_data_helper, pd.DataFrame)
        assert len(cat_columns) > 0
        in_data_enc = encoder_helper(in_data_helper, cat_columns, in_data_helper['Churn'])
        assert len(in_data_enc.columns) < len(in_data_helper.columns)
        logging.info("Encoder check: SUCCESS")
    except AssertionError as err:
        logging.error("Number of columns not matching ...")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, in_data_eng):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(in_data_eng)
        assert x_train.shape[0] + x_test.shape[0] == in_data_eng.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == in_data_eng.shape[0]
        logging.info("Splitting the data into train and test: SUCCESS")
    except AssertionError as err:
        logging.error("Splitting not done properly")
        raise err


def test_train_models(train_models, in_data):
    '''
    test train_models
    '''
    try:
        perform_eda(in_data)
        cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                       'Income_Category', 'Card_Category']
        in_data = encoder_helper(in_data, cat_columns, in_data['Churn'])
        x_train, x_test, y_train, y_test = perform_feature_engineering(in_data)
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
            x_train, x_test, y_train, y_test)
        path_images = r'./images/results/'
        list_results_files = [
            'feature_importances.png',
            'logistic_results.png',
            'rf_results.png',
            'roc_curve_result.png']
        check_files(list_results_files, path_images)
        logging.info(f"Files in the {path_images} path validated: SUCCESS")
        path_models = f'./models/'
        list_models = ['logistic_model.pkl', 'rfc_model.pkl']
        check_files(list_models, path_models)
        logging.info(f"Files in the {path_models} path validated: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Files not matching ...")
        raise err


if __name__ == "__main__":
    in_data = test_import(import_data)
    test_eda(perform_eda, in_data)
    test_encoder_helper(encoder_helper, in_data)
    test_perform_feature_engineering(perform_feature_engineering, in_data)
    test_train_models(train_models, in_data)
