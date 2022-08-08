# library doc string


# import libraries

import os
os.environ['QT_QPA_PLATFORM']='offscreen'
# import shap
import joblib
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report




def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            in_data: pandas dataframe
    '''
    try:
        in_data = pd.read_csv(pth) #pulling the bank data
        print ("SUCCESS: File read correctly ...")
        return in_data
    except FileNotFoundError:
        print ("Error: File not find in the given path ...")
 

def plot_and_save(in_data, col):
    plt.figure(figsize = (20,10))
    if col in ['Churn', 'Customer_Age']:
        in_data[col].hist()
        print (f"Saving the {col} distribution plot to /images/eda folder ...")
        plt.savefig(f'./images/eda/{col}_distribution.png')
    elif col in ['Marital_Status']:
        in_data[col].value_counts('normalize').plot(kind='bar')
        print (f"Saving the {col} distribution plot to /images/eda folder ...")
        plt.savefig(f'./images/eda/{col}_distribution.png')
    elif col in ['Total_Trans_Ct']:
        sns.histplot(in_data['Total_Trans_Ct'], stat='density', kde=True)
        print (f"Saving the total transaction distribution plot to /images/eda folder ...")
        plt.savefig(f'./images/eda/total_transaction_distribution.png')
    else:
        sns.heatmap(in_data.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        print (f"Saving the heatmap to /images/eda folder ...")
        plt.savefig(f'./images/eda/heatmap.png')
    

def perform_eda(in_data):
    '''
    perform eda on in_data and save figures to images folder
    input:
            in_data: pandas dataframe

    output:
            None
    '''
    cat_columns = ['Gender','Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'] #cat columns
    quant_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'] #quant columns
    in_data['Churn'] = in_data['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1) #creating the Churn column
    for col in ['Churn', 'Customer_Age', 'Marital_Status','Total_Trans_Ct', '_']:
        plot_and_save(in_data, col) # plotting and saving the columns
    

def encoder_helper(in_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            in_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            in_data: pandas dataframe with new columns for
    '''
    print ('Features renaming ...')
    for col in category_lst:
        col_lst = []
        group = in_data.groupby(col).mean()['Churn']
        for val in in_data[col]:
            col_lst.append(group.loc[val])
        in_data[f'{col}_Churn'] = col_lst
    keep_cols = [col for col in in_data.columns if col.split('_')[-1] == 'Churn']
    in_data = in_data[keep_cols]
    return in_data


def perform_feature_engineering(in_data):
    '''
    input:
              in_data: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print ('Splitting input data into training and testing ...')
    y, X = in_data.loc[:,'Churn'], in_data.loc[:,[col for col in in_data.columns if col != 'Churn']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    print ("Saving RF results ...")
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.9, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.04, str(classification_report(y_train, y_train_preds_rf, zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf, zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'./images/results/rf_results.png')
    
    print ("Saving LR results ...")
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.9, str('Logistic Regression results'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.04, str(classification_report(y_train, y_train_preds_lr, zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr, zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace    plt.axis('off')
    plt.axis('off')
    plt.savefig(f'./images/results/logistic_results.png')

   

    


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(output_pth + 'feature_importances.png')
   

    
def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print ("Performing a grid search ...")
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']}
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    X_data = pd.concat([x_train, x_test], ignore_index = True)
    lrc.fit(x_train, y_train)
    feature_importance_plot(cv_rfc, X_data, './images/results/')
    print ("Plotting ROC curves ...")
    plot_models(lrc, cv_rfc, x_test, y_test)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    print ("Saving the best estimator for Random Forest ...")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    print ("Saving the best estimator for Logistic Regression ...")
    joblib.dump(lrc, './models/logistic_model.pkl')
    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr
    

def plot_models(model1, model2, x_test, y_test):
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(model1, x_test, y_test)
    ax = plt.gca()
    # lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(model2.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(f'./images/results/roc_curve_result.png')
    

if __name__ == "__main__":
    in_data = import_data(r"./data/bank_data.csv")
    perform_eda(in_data)
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    in_data = encoder_helper(in_data, cat_columns, in_data['Churn'])
    x_train, x_test, y_train, y_test = perform_feature_engineering(in_data)
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(x_train, x_test, y_train, y_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

 
    