"""
Python script file to convert notebook ml model into reproducible and clea code.

Date: 29, April 2022
Author: Kaushal Kishore
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth: object) -> object:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    input_df = pd.read_csv(pth, sep=',', index_col=0)
    return input_df


def perform_eda(data):
    """

    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    print(data.head())
    # create churn binary column using list comprehension
    data['Churn'] = [
        0 if x == 'Existing Customer' else 1 for x in data['Attrition_Flag']]
    plt.figure(figsize=(16, 8))
    data['Churn'].hist()
    # save eda plots in images/eda folder
    plt.savefig('./images/eda/churn_distribution.png')
    data['Customer_Age'].hist()
    plt.savefig('./images/eda/age_distribution.png')
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution.png')
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_transaction_distribution.png')
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(input_data, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for n
            aming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    # get the list of all categorical column
    category_lst = input_data.columns[input_data.dtypes == 'object']
    for cat in category_lst:
        var_groups = input_data.groupby(cat).mean()[response]
        input_data[cat + '_churn'] = [var_groups.loc[x] for x in df[cat]]
    return input_data


def perform_feature_engineering(input_data, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    x_input = input_data[input_data.columns[input_data.dtypes
                                            != 'object'][1:]].drop(response, axis=1)
    y_output = input_data[response]
    x_train, x_test, y_train_df, y_test_df = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return x_input, y_output, x_train, x_test, y_train_df, y_test_df


def classification_report_image(train_y,
                                test_y,
                                train_preds_lr,
                                train_preds_rf,
                                test_preds_lr,
                                test_preds_rf):
    """
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
    """
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(test_y, test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(train_y, train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rfc_report.png')
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(train_y, train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(test_y, test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/lrc_report.png')


def feature_importance_plot(ml_model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = ml_model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_imp.png')


def train_models(in_train, in_test, out_train, out_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(in_train, out_train)
    # Logistics regression classification
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(in_train, out_train)
    preds_y_train_rf = cv_rfc.best_estimator_.predict(in_train)
    preds_y_test_rf = cv_rfc.best_estimator_.predict(in_test)
    preds_y_train_lr = lrc.predict(in_train)
    preds_y_test_lr = lrc.predict(in_test)
    plt.figure(figsize=(12, 8))
    lrc_plot = plot_roc_curve(lrc, in_test, out_test)
    plt.savefig('./images/results/lrc_roc.png')
    # Model comparision plot
    # plots
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        in_test,
        out_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/results/lrc_rfc_roc.png')
    # save the model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    return cv_rfc, preds_y_train_rf, preds_y_test_rf, preds_y_train_lr, preds_y_test_lr


if __name__ == "__main__":
    df = import_data('./data/bank_data.csv')
    perform_eda(df)
    en_df = encoder_helper(df, 'Churn')
    X, y, X_train, X_test, y_train, y_test = perform_feature_engineering(
        en_df, 'Churn')
    model, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = \
        train_models(X_train, X_test, y_train, y_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(model, x_data=X, output_pth='./images/results/')
