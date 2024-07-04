#All imports

import warnings
import seaborn as sns
import pylab as py
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import duckdb as db
import dtreeviz
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from skimpy import skim
from scipy.stats import shapiro
from scipy.stats import pointbiserialr
from scipy.stats import fisher_exact
from plotly.subplots import make_subplots

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    LabelEncoder,
    KBinsDiscretizer,
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    make_scorer,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,RandomForestRegressor,
)
from sklearn import svm
from sklearn import metrics


#Define scoring metrics 
scoring_metrics = ["accuracy", "precision", "recall", "f1"]





def binary_formatter(df, switch):
    """Formats binary variables into yes/no notation,booleans, or reverts dataframe to original format.

    Args:
        df (pandas): overall travel insurance dataframe
        switch (string): specifies output format, must be "binary", "string", or "revert"

    Returns:
        df (pandas): input df in desired format
    """
    if switch == "binary":
        df.replace(to_replace="Yes", value=1, regex=False, inplace=True)
        df.replace(to_replace="No", value=0, regex=False, inplace=True)
    elif switch == "string":
        df.replace(to_replace=1, value="Yes", regex=False, inplace=True)
        df.replace(to_replace=0, value="No", regex=False, inplace=True)
    elif switch == "revert":
        df = (
            pd.read_csv("TravelInsurancePrediction.csv")
            .rename(
                columns={
                    "": "index",
                    "Age": "age",
                    "Employment Type": "employment",
                    "GraduateOrNot": "college_grad",
                    "AnnualIncome": "income",
                    "FamilyMembers": "family_size",
                    "ChronicDiseases": "conditions",
                    "FrequentFlyer": "frequent_traveler",
                    "EverTravelledAbroad": "first_trip",
                    "TravelInsurance": "travel_insurance",
                }
            )
            .iloc[:, 1:]
        )
    else:
        print("choose either binary, string, or revert as the switch option.")
    return df

def binary_grapher(df,column, title="", labels=[None, None], file_type=None):
    """Graphs binary variables as histogram with specified title, labels, and file type using plotly

    Args:
        column (pandas series): desired column to graph
        title (str, optional): desired title for graph. Defaults to ''.
        labels (list, optional): list of strings in [x,y] format, specifies the axis labels. Defaults to [None,None].
        file_type (string, optional): specifies 'svg' or 'png' for plotly renderers. Defaults to None.
    """
    fig = px.histogram(
        binary_formatter(df, "string"),
        x=column,
        color=column,
        text_auto=True,
        labels={1: "Yes", 0: "No"},
        title=title,
        color_discrete_sequence={0: "rgb(154, 211, 189)", 1: "rgb(45, 105, 123)"},
        category_orders={column: ["No", "Yes"]},
    )
    fig.update_layout(
        width=1200,
        height=500,
        bargap=0.05,
    )
    if labels:
        fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])
    fig.show(file_type)
    
def create_heatmap(df, reversed=True):
    """Creates a heatmap of correlations

    Args:
        df (dataframe): correlation matrix
        reversed (bool, optional): _description_. Defaults to True.
    """
    cmap = sns.color_palette("crest", as_cmap=True).reversed(
        sns.color_palette("dark:#5A9_r", as_cmap=True)
    )
    if reversed == False:
        cmap = sns.color_palette("crest", as_cmap=True)
    sns.heatmap(
        df,
        annot=True,
        cmap=cmap,
    )
    plt.show()

def range_grapher(column, title="", labels=[None, None], file_type=None):
    """Create bar graph of non-binary numerical variables

    Args:
        column (string): name of column in df to graph
        title (str, optional): desired graph title. Defaults to ''.
        labels (list, optional): list of strings in [x,y] format, specifies the axis labels. Defaults to [None,None].
        file_type (_type_, optional): _descrispecifies 'svg' or 'png' for plotly renderers. Defaults to None. Defaults to None if interactive plot desired.
    """
    # Calculate prevalence of each column
    tallies = df[column].value_counts()

    # Plot
    fig = px.bar(
        tallies,
        x=tallies.index,
        y=tallies.values,
        color=tallies,
        color_continuous_scale="Darkmint",  # "Emrld",#"BlueRed",
        title=title,
        text_auto=True,
    )
    fig.update_layout(width=1200, height=500, bargap=0.05)
    if labels:
        fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])

    fig.show(file_type)

def make_subplot(df,figure, feature, position):
    """Makes bar subplot for row and column of figure specified

    Args:
        figure (plotly go): Plotly graph_objects figure
        feature (string): the name of the column within pandas df to make plot of
        position (list): list of integers in [row,column] format for specifying where in figure to plot graph
        labels (list, optional): Title, xlabel, and ylabel for subplots. Defaults to ['',None,None].
    """
    tallies = df[feature].sort_values(ascending=True).value_counts()
    figure.add_trace(
        go.Bar(
            x=tallies.index,
            y=tallies.values,
            name="",
            marker=dict(color=["rgb(154, 211, 189)", "rgb(45, 105, 123)"]),
            hovertemplate="%{x} : %{y}",
            text=tallies.values,
        ),
        row=position[0],
        col=position[1],
    )
    figure.update_layout(bargap=0.2)

def numerical_subplot(df,fig, column, position_list):
    """Function creates bargraph of numerical continuous feature and adds to trace of plotly figure

    Args:
        fig (plotly figure): Figure to hold subplot
        column (df.series): feature of dataset
        position_list (list): list in [row,col] format
    """
    tallies = df[column].value_counts()
    fig.add_trace(
        go.Bar(
            x=tallies.index,
            y=tallies.values,
            name=str.capitalize(column),
            hovertemplate="%{x} : %{y}",  # ,text=tallies.values,
            marker=dict(
                color=tallies.index,
                colorscale="DarkMint",
            ),
            textposition="inside",
        ),
        row=position_list[0],
        col=position_list[1],
    )
    fig.update_layout(bargap=0.2)

def calculate_log_model_stats(y_actual, y_pred):
    """Calculates accuracy, precision, recall, and f1 score for a logistic regression model

    Args:
        y_actual (array/df): y training or test set data
        y_pred (array): predictions from model.predict method in sklearn

    Returns:
        list: list of metrics
    """
    training_accuracy = accuracy_score(y_actual, y_pred)
    training_precision = precision_score(y_actual, y_pred)
    training_recall = recall_score(y_actual, y_pred)
    training_f1 = f1_score(y_actual, y_pred)
    return [training_accuracy, training_precision, training_recall, training_f1]

def create_contingency_table(df,column1, column2):
    """Calculates contingency matrix for two variables within binary_df

    Args:
        column1 (str): first variable
        column2 (str): second variable

    Returns:
        contingency table in pandas dataframe format
    """
    matrix = pd.crosstab(df[column1], df[column2], margins=False)
    return matrix


def fishers_exact_test(matrix):
    """Function conducts fisher's exact test on two binary variables via the contingency table between them

    Args:
        matrix (pandas df): a contingency table

    Returns:
        observed_odds_ratio: the observed odds_ratio in the dataset, which is assumed to be 1 under the null hypothesis
        p_value: p_value of seeing an equally extreme value under null hypothesis
    """
    observed_odds_ratio, p_value = fisher_exact(matrix)
    return observed_odds_ratio, p_value

def create_fishers_array(binary_df):
    """Creates array of fisher's exat test p-values"""
    # Use binary_df columns as the variable set
    variables = binary_df.columns
    # Create an all-zero dataframe to repopulate with p-values
    array = pd.DataFrame(
        np.zeros((len(variables), len(variables)), dtype=int), columns=variables
    )

    # Create contingency matrix and fishers_exact pvalues for all variable combinations in the set
    for column in variables:
        array[column] = [
            round(fishers_exact_test(create_contingency_table(binary_df,column, x))[1], 3)
            for x in variables
        ]
    array.set_index(variables, inplace=True)
    return array 

def create_confusion_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    return pd.DataFrame(
        data=cm,
        columns=["Actual Positive", "Actual Negative"],
        index=["Predicted Positive", "Predicted Negative "],
    )

def create_model_predictions(model, x):
    """Predicts labels based on model

    Args:
        model (sklearn classifier): fitted classifier 
        x (dataframe): x_train or x_test partition

    Returns:
        predictions: np.array of model predictions
    """
    return model.predict(x)

def conduct_cross_validation(model,x_train,y_train):
    """Conducts cross validation on specified model

    Args:
        model (sklearn model instance): model to implement kfold cross-validation on

    Returns:
        dataframe: df of means, stds, and confidence intervals
    """

    kfold = KFold(n_splits=10)
    scoring_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    # Initialize lists
    means = []
    stds = []
    intervals = []

    for metric in scoring_metrics:

        # Create vector of evaluation scores
        cv_vector = cross_val_score(
            estimator=model,
            X=x_train,
            y=y_train,
            cv=kfold,
            scoring=metric,
            # error_score="raise",
        )

        cv_mean = cv_vector.mean()
        cv_std = cv_vector.std()

        means.append(cv_mean)
        stds.append(cv_std)
        intervals.append([(cv_mean - cv_std).round(3), (cv_mean + cv_std).round(3)])

    df = pd.DataFrame(
        data={
            "mean_score": means,
            "standard_deviation": stds,
            "confidence_interval": limit_percentages(intervals),
        },
        index=scoring_metrics,
    )
    return df

def calculate_model_statistics(cm, column="values"):
    """Calculate accuracy, precision, recall, and f1 for a given confusion matrix

    Args:
        cm (sklearn confusion matrix): confusion matrix of y_actual and y_pred

    Returns:
        dataframe: pandas dataframe with metrics
    """
    from sklearn.metrics import confusion_matrix

    # Isolate tallies by category
    true_positives = cm[0, 0]
    false_negatives = cm[0, 1]
    false_positives = cm[1, 0]
    true_negatives = cm[1, 1]

    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return pd.DataFrame(
        data={column: [accuracy, precision, recall, f1]},
        index=["accuracy", "precision", "recall", "f1"],
    )

def shapiro_wilk(feature):
    """Intakes feature to perform Shapiro-Wilk test of normality for that given data series

    Args:
        feature (pandas series): dataframe column representing a given feature
    """
    statistics = shapiro(feature)  # , axis=0)

    name = ["SP-test statistic", "p-value"]
    return pd.DataFrame(data={feature.name: name, "values": statistics}).round(4)

def show_logistic_model_features(x_training_set, logistic_model):
    """Pulls dictionary of feature names and logistic model coefficients from fitted sklearn model

    Args:
        x_training_set (pandas_df): x dataframe used to train the model
        logistic_model (sklearn LogisticRegression instance): model instance

    Returns:
        model_features: dictionary of model features and model coefficients
    """
    features = list(x_training_set.columns)
    coefs = logistic_model.coef_[0]
    model_features = {features[i]: coefs[i] for i in range(len(features))}
    return model_features

def conduct_grid_search_tuning(model, grid,X,Y):
    """Conducts gridsearch for specified model and hyperparameter settings

    Args:
        model (string): string specifying model to test, must be 'knn', 'logistic_regression','decision_tree', or 'random_forest'
        grid (dictionary): grid of lists specifying options for hyperparameters to tune
        xy (list): x and y for model fitting, should be in [x_train,y_train] format
    """
    models = {
        "knn": KNeighborsClassifier(),
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "bagging": BaggingClassifier(),
        "nb": CategoricalNB(),
    }

    classifier = models[model]
    grid_search = GridSearchCV(classifier, grid, cv=10)
    grid_search.fit(X, Y)

    best_params = grid_search.best_params_
    model_accuracy = np.round(grid_search.best_score_, 4)

    return best_params, model_accuracy

def consolidate_model_stats(model_summaries):
    model_names=[
            "logistic_regression",
            "k_nearest_neighbors",
            "decision_tree",
            "naive_bayes",]
    df=pd.DataFrame(data=np.zeros([4,4]),
        columns=model_names,
        index=scoring_metrics,
    )
    for i in range(0,len(model_summaries)):
        df[model_names[i]]=model_summaries[i].iloc[:,1]
    return df

def compare_models(df1,df2,df3,df4):
    """Function prints all tuned model training set performances in a single df

    Returns:
        all_models: training set performance of single and ensemble models
    """
    #Isolate training set performance of each ensemble model 
    rf = df2.iloc[:, 1]
    sclf = df3.iloc[:, 0]
    vc = df4.iloc[:, 0]

    #Concatenate single models as well 
    all_models = pd.concat([df1,rf, sclf, vc], axis=1)
    all_models.columns = list(df1.columns)+["random_forest", "stacked_classifier", "voting_classifier"]
    return all_models

def resplit_data(X, Y, test_size=0.3):
    """Reexecute train-test-split

    Args:
        X (df): features
        Y (df): labels
        test_size (float, optional): Specify proportion of test_set. Defaults to 0.3.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        stratify=Y,  # random_state=1
    )
    return x_train, x_test, y_train, y_test

def calculate_v2_model_training_stats(model, x_training, y_training):
    """Executes model predictions, calculates confusion matrix, and outputs model statistics for the training set

    Args:
        model (classifier): sklearn classifier
        x_training (df): x training set 
        y_training (df ): y training set 

    Returns:
        performance: dataframe of accuracy, precision, recall, and F1 score of model predictions
    """
    # Calculate predictions
    ypred = create_model_predictions(model, x_training)

    # Calculate confusion matrix
    cm = confusion_matrix(y_training, ypred)

    # Calculate model statistics
    performance = calculate_model_statistics(cm, column="v2: training")
    return performance

def calculate_v2_model_testing_stats(model, x_testing, y_testing):
    """Executes model predictions, calculates confusion matrix, and outputs model statistics for the testing set

    Args:
        model (classifier): sklearn classifier
        x_testing (df): x testing set 
        y_testing (df ): y testing set 

    Returns:
        performance: dataframe of accuracy, precision, recall, and F1 score of model predictions
    """
    # Calculate predictions
    ypred = create_model_predictions(model, x_testing)

    # Calculate confusion matrix
    cm = confusion_matrix(y_testing, ypred)

    # Calculate model statistics
    performance = calculate_model_statistics(cm, column="v2: testing")
    return performance

def test_score_range(df):
    """Calculates the interval of test_scores within the grid_search_results output

    Args:
        df (dataframe): gridsearchCV results in pandas dataframe form

    Returns:
        range: list of min and max score values
    """
    return df.apply(lambda row: [np.round(row.min(),3), np.round(row.max(),3)], axis=1)

def assemble_performance_metrics_df(cross_val_df, v2_train_stats, v2_test_stats):
    """Assemble model performance df across cross-val and v2 models

    Args:
        cross_val_df (df): dataframe of cross-validation performance
        v2_train_stats (df): dataframe of scoring metrics for v2 model on the training set
        v2_test_stats (df): dataframe of scoring metrics for v2 model on the testing set

    Returns:
        output_df: performance dataframe for both models across cv, train, and test sets
    """
    output_df = pd.concat(
        [cross_val_df.iloc[:, 0].copy(), v2_train_stats, v2_test_stats], axis=1
    ).rename(columns={"mean_score": "v1: cross_val"})
    return output_df

def limit_percentages(intervals):
    """Function caps confidence interval for percentages at 1, and floors it 0"""
    for interval in intervals:
        if interval[0] < 0:
            interval[0] = 0
        if interval[1] > 1:
            interval[1] = 1.000
    return intervals

def test_score_range(df):
    return df.apply(lambda row: [np.round(row.min(),3), np.round(row.max(),3)], axis=1)

def calculate_grid_search_ranges(df,metrics):
    """Calculates range of test_scores for all metrics specified

    Args:
        metrics (list): list of strings denoting metrics

    Returns:
        output: dataframe of relevant columns from overall gridsearch output
    """
    output = pd.DataFrame()
    for metric in metrics:
        pattern = "split\d_test_" + metric
        data = df.filter(regex=pattern)
        column_name = metric + "_range"
        output[column_name] = test_score_range(data)
    return output

def pull_rank_one_runs(grid_results, score):
    """Pulls rank==1 runs for score specified from gridsearch results"""
    rank = "rank_test_" + score
    columns = ["mean_test_" + score, "std_test_" + score]
    output = grid_results[columns][grid_results[rank] == 1]
    output["tree_number"] = output.index
    output = output.iloc[:, [2, 0, 1]]
    # output.set_index('tree_number',inplace=True)
    return output  # .style.format().hide(axis='index')

def restrict_grid_search_table(grid_search_results_df, metrics, grid_search_ranges):
    """Function takes grid_search printout and filters down to metrics-related columns and metrics-ranges
    Args:
        grid_search_results_df (df): complete output of grid search
        metrics (list): list of strings matching metrics of interest
        grid_search_ranges (df): dataframe with (min,max) ranges for each metric in metrics list

    Returns:
        dataframe: df with metrics and metrics ranges
    """
    # Define regex based on metrics list of strings
    regex = "|".join(metrics)
    columns = grid_search_results_df.filter(regex=regex).columns
    output = grid_search_results_df[columns]
    # output=pd.concat([output,grid_search_ranges],axis=1)
    return output