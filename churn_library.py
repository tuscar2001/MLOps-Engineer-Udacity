###Author: Oscar Adimi 
###Date: 8/8/2022 
###Description: This file ia aimed at building a ML pipeline for customers that are 
### likely to churn. The steps of the pipeline are: 1. Churn data collection 
###                                                 2. Preliminary Checks 
###                                                 3. Splitting the data 
###                                                 4. Training the data  
### Some code outputs have been saved in the appropriate paths.           


# import libraries
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
 
def plot_and_save(in_data_plt, col):
    """
    Plotting and saving the plot results
    """
    plt.figure(figsize = (20,10))
    if col in ['Churn','Customer_Age']:
        in_data_plt[col].hist()
        print (f"Saving the {col} distribution plot to /images/eda folder ...")
        plt.savefig(f'./images/eda/{col}_distribution.png')
    elif col in ['Marital_Status']:
        in_data_plt[col].value_counts('normalize').plot(kind='bar')
        print (f"Saving the {col} distribution plot to /images/eda folder ...")
        plt.savefig(f'./images/eda/{col}_distribution.png')
    elif col in ['Total_Trans_Ct']:
        sns.histplot(in_data_plt['Total_Trans_Ct'],stat='density',kde=True)
        print ("Saving the total transaction distribution plot to /images/eda folder ...")
        plt.savefig('./images/eda/total_transaction_distribution.png')
    else:
        sns.heatmap(in_data_plt.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        print ("Saving the heatmap to /images/eda folder ...")
        plt.savefig('./images/eda/heatmap.png')
    

def perform_eda(in_data_pfr):
    '''
    perform eda on in_data and save figures to images folder
    input:
            in_data: pandas dataframe

    output:
            None
    '''
    #creating the Churn column
    in_data['Churn'] = in_data_pfr['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    for col in ['Churn','Customer_Age','Marital_Status','Total_Trans_Ct','_']:
        plot_and_save(in_data_pfr, col) # plotting and saving the columns
    

def encoder_helper(in_data_hp, cat_lst, num_list, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            in_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
             be used for naming variables or index y column]

    output:
            in_data: pandas dataframe with new columns for
    '''
    print ("Dummy categorical freatures ...")
    cat_df = pd.get_dummies(in_data_hp[cat_lst])
    num_df = in_data[num_list]
    outdata = pd.concat([cat_df,num_df],axis = 1)
    outdata['Churn'] = response
    return outdata

def perform_feature_engineering(in_data_eng):
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
    y_dsn, x_dsn = in_data_eng.loc[:,'Churn'], in_data_eng.loc[:,[col for col in in_data_eng.columns if col != 'Churn']]
    x_tr, x_ts, y_tr, y_ts = train_test_split(x_dsn,y_dsn,test_size= 0.3,random_state=42)
    print (x_tr.shape,y_tr.shape,x_ts.shape,y_ts.shape)
    return x_tr,x_ts,y_tr,y_ts

def classification_report_image(y_tr,
                                y_ts,
                                y_tr_preds_lr,
                                y_tr_preds_rf,
                                y_ts_preds_lr,
                                y_ts_preds_rf):
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
    plt.text(0.01, 0.9, str('Random Forest Train'),{'fontsize': 10},fontproperties = 'monospace')
    plt.text(0.01, 0.04, str(classification_report(y_tr,y_tr_preds_rf,zero_division=0)),{'fontsize': 10},fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),{'fontsize': 10}, fontproperties = 'monospace')
     # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str(classification_report(y_ts,y_ts_preds_rf,zero_division=0)), {'fontsize': 10},fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    
    print ("Saving LR results ...")
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,0.9,str('Logistic Regression results'),{'fontsize': 10},fontproperties = 'monospace')
    plt.text(0.01,0.04,str(classification_report(y_tr,y_tr_preds_lr,zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01,0.6,str('Logistic Regression Test'),{'fontsize': 10},fontproperties = 'monospace')
    plt.text(0.01,0.7,str(classification_report(y_ts,y_ts_preds_lr,zero_division=0)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace    plt.axis('off')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
   
def feature_importance_plot(model,x_data,output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_importances.png')
   
    
def train_models(x_tr, x_ts, y_tr, y_ts):
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
    'n_estimators': [100, 200, 500, 700],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [3, 4, 5, 100],
    'criterion' :['gini', 'entropy']}

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_tr, y_tr)
    x_data = pd.concat([x_tr, x_ts], ignore_index = True)
    lrc.fit(x_train, y_train)
    feature_importance_plot(cv_rfc, x_data, './images/results/')
    print ("Plotting ROC curves ...")
    print ("RF best model {}".format(cv_rfc.best_estimator_))
    print ("Logistic Regression best model: {}".format(lrc))
    plot_models(lrc, cv_rfc, x_ts, y_ts)
    y_tr_preds_rf = cv_rfc.best_estimator_.predict(x_tr)
    y_ts_preds_rf = cv_rfc.best_estimator_.predict(x_ts)
    y_tr_preds_lr = lrc.predict(x_tr)
    y_ts_preds_lr = lrc.predict(x_ts)
    print ("Saving the best estimator for Random Forest ...")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    print ("Saving the best estimator for Logistic Regression ...")
    joblib.dump(lrc, './models/logistic_model.pkl')
    return y_tr_preds_rf, y_ts_preds_rf, y_tr_preds_lr, y_ts_preds_lr
    

def plot_models(model1, model2, x_plot, y_plot):
    """
    creating models plots and saving the results
    """
    #plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(model1, x_plot, y_plot)
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    # lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(model2.best_estimator_, x_plot, y_plot, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax= a_x, alpha=0.8)
    rfc_disp.plot(ax = a_x, alpha = 0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    

if __name__ == "__main__":
    # import shap
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import plot_roc_curve, classification_report
    in_data = import_data(r"./data/bank_data.csv")
    perform_eda(in_data)
    cat_columns = ['Gender','Education_Level','Marital_Status',
                    'Income_Category','Card_Category']


    quant_columns = ['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count', 
        'Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal',
        'Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio']
    in_data = encoder_helper(in_data, cat_columns, quant_columns, in_data['Churn'])
    x_train, x_test, y_train, y_test = perform_feature_engineering(in_data)
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(x_train,x_test,y_train,y_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

 
    