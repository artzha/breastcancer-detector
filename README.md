# breastcancer-detector

Doctors frequently have to diagnose breast cancer and determine if it is malignant or benign. I trained this model
for doctors to quickly determine the condition of a patient's breast cancer. I used a breast cancer dataset
from Wisconsin with 569 diagnoses.

## Selecting Data

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/training-data.png "network_structure")

Upon The dataset consists of 32 relevant features for each diagnoses as well as if it was malignant of benign.

## Dataset Analysis

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/Diagnosis-Distribution.png  "network_structure")

Looking at malignant v benign distribution, we have more benign tumor diagnoses than malignant diagnoses.
This could potentially skew our model towards benign evaluations. Therefore, it is important to select a regression model
that minimizes this skewing.

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/feature-comparison.png  "network_structure")

I also plotted the other features on various scatterplots to look for any abnormalities or potentially dangerous outliers.
Looking at the data, it seems like all of the features are relatively consistent with each other. Therefore, I don't
need to specially adjust my dataset for these features.

# Choosing the Classification Model

## Logistic Regression

I used a logistic regression model to model the data first, a common regression model for binary outputs and
categorical data. I chose not to use all 32 features for risk of overfitting my model to this dataset.

Features Included

1. radius_mean
2. perimeter_mean
3. area_mean
4. compactness_mean
5. concave points_mean

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/Logistic-Stats.png  "network_structure")

Even though my model performed with *89.279% accuracy*, it's cross validation scores were noticeably lower that I would like.
Because of this, I decided to experiment with using other classification models.

## Random Forest

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/RandomForestDiagram.png  "network_structure")

I chose to use a Random Forest Classification model next because it can accurately handle large amounts of input
features and can balance error in class population with different sizes. This is important because my dataset consists
of more benign than malignant diagnoses. In addition, this model is designed to avoid overfitting data and be very efficient.

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/RandomForest-11fet.png  "network_structure")

Even though this model had higher accuracy and cross-validation scores, it can be further improved. The random forest model
generates the importance of each feature that is fed into the model.

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/feature-weights.png  "network_structure")

Based on these weights, I chose to only input the most important features into my new model: concave points_mean,
area_mean, radius_mean, perimeter_mean, concavity_mean. By reducing the number of input features, I can generalize my
model, which makes it more accurate for evaluating new data.

![network structure](https://github.com/KingArthurZ3/breastcancer-detector/blob/master/rsc/RandomForest-5fet.png  "network_structure")

By only considering the five most important features, my model's accuracy rose to *96.134%* with higher cross-validation
scores than previously acheived. However, there's still room for improvement, so I will continue to experiment with other
regression models and techniques.

