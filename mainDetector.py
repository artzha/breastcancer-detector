# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

cancerData = pd.read_csv('data/data.csv')
features_mean = list(cancerData.columns[2:11])

trainData, testData = train_test_split(cancerData, test_size = 0.3)

def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data[outcome]):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be referred outside the function:
    model.fit(data[predictors], data[outcome])

# logistic regression model
# predictor_var = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean']
# outcome_var = 'diagnosis'
# model = LogisticRegression()
# classification_model(model, trainData, predictor_var, outcome_var)

#Random Forest Classifier all 11 features
#Use all the features of the nucleus
# predictor_var = features_mean
# outcome_var = 'diagnosis'
# model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
# classification_model(model, cancerData, predictor_var, outcome_var)

#Create a series with feature importances:
# featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
# print(featimp)

# Random Forest Model using top 5 features
predictor_var = ['concave points_mean', 'area_mean', 'radius_mean', 'perimeter_mean', 'concavity_mean']
outcome_var = 'diagnosis'
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, cancerData, predictor_var, outcome_var)