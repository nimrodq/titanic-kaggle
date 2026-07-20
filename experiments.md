# Titanic Model Experiments

This document records the experiments conducted during development of the Titanic survival prediction model.


# Experiment 1 - Baseline Model

## Objective

Create a simple baseline model using basic passenger information.

## Model

Logistic Regression

## Features

Initial features:

- Pclass
- Sex
- Age
- Fare
- Embarked


## Result

Validation accuracy was approximately:

```text
78%
```

The baseline model performed reasonably but struggled to capture nonlinear passenger relationships.


---

# Experiment 2 - Feature Engineering

## Objective

Improve model performance by creating additional passenger-based features.


## Added Features

### Family Features

Added:

- FamilySize
- FamilySizeGroup
- IsAlone


### Passenger Features

Added:

- Title extraction
- Age imputation by title


### Cabin Features

Added:

- Deck
- CabinKnown


### Fare Features

Added:

- FarePerPerson
- FareBand


### Ticket Features

Added:

- TicketGroupSize


## Result

Feature engineering improved model performance by providing more meaningful passenger relationships.


---

# Experiment 3 - Model Comparison

## Objective

Compare different machine learning algorithms.


## Models Tested


## Random Forest

Advantages:

- Handles nonlinear relationships
- Robust against noise

Validation Accuracy:

```text
~79%
```


## XGBoost

Advantages:

- Gradient boosting approach
- Strong performance on structured datasets

Validation Accuracy:

```text
~81.5%
```


## LightGBM

Advantages:

- Efficient gradient boosting
- Fast training

Validation Accuracy:

```text
~81%
```


## CatBoost

Advantages:

- Strong categorical feature handling
- Effective on small datasets

Validation Accuracy:

```text
82.1%
```


CatBoost achieved the best validation performance.


---

# Experiment 4 - Hyperparameter Tuning

## Objective

Improve model performance through parameter optimisation.


## Method

RandomizedSearchCV with cross validation.


## Tuned Parameters


### CatBoost

- iterations
- depth
- learning rate
- L2 regularisation


### XGBoost

- number of estimators
- max depth
- learning rate
- subsampling


### LightGBM

- number of leaves
- depth
- regularisation


### Random Forest

- number of estimators
- maximum depth
- minimum samples


## Result

Best tuned model:

```text
CatBoostClassifier
```

Validation accuracy:

```text
82.1%
```


---

# Experiment 5 - Cross Validation

## Objective

Verify that validation performance was not caused by a lucky train-validation split.


## Method

10-Fold Stratified Cross Validation


## Result

```text
Mean Accuracy: 82.6%
```


The result confirmed that the model consistently achieves around 82% accuracy.


---

# Experiment 6 - Feature Leakage Investigation

## Problem

Initial experiments showed:

Validation:

```text
82.6%
```

Kaggle:

```text
77.5%
```


A generalisation issue was suspected.


## Removed Features


## SurnameCount

Previous implementation:

```text
Combined train and test surname counts
```

This used information from the hidden test distribution.


## TicketGroupSize

Previous implementation:

```text
Combined train and test ticket counts
```

This also used test distribution information.


## Fix

Changed statistical calculations to use training data only.


Example:

Before:

```text
train + test ticket counts
```

After:

```text
training ticket counts only
```


## Expected Impact

The model should generalise better to unseen Kaggle test data.


---

# Current Best Model

| Metric | Score |
|---|---:|
| Validation Accuracy | 82.1% |
| 10-Fold CV Accuracy | 82.6% |
| Kaggle Public Score | 77.99% |


# Future Experiments

## Family Survival Features

Planned feature engineering:

- Same surname survival
- Same ticket survival
- Family member outcomes


## Ensemble Methods

Potential improvements:

- Voting Classifier
- Stacking Classifier

## Feature Importance Analysis

Permutation importance was performed on the final Logistic Regression model.

The most influential feature was WomanChild, which combines gender and age information to capture the historical "women and children first" survival pattern.

Other important features included:

- FamilySizeGroup
- Sex
- Age
- Fare
- Passenger Title

Some engineered features such as SurnameCount, TicketGroupSize and FarePerPerson showed negative importance, suggesting they introduced additional noise rather than improving prediction performance.


## Explainability

Future analysis:

- SHAP values
- Misclassification analysis