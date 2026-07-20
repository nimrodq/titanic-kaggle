# Titanic - Machine Learning from Disaster

A machine learning project that predicts passenger survival on the Titanic dataset from Kaggle.

The project focuses on feature engineering, model comparison, hyperparameter tuning, and improving generalisation performance on a small tabular dataset.

## Project Overview

The objective is to predict whether a passenger survived the Titanic disaster using information such as:

- Passenger class
- Gender
- Age
- Fare
- Family relationships
- Ticket information
- Cabin information
- Passenger titles

## ML Pipeline


Raw Titanic Dataset
|
v
Data Validation
|
v
Feature Engineering
|
v
Preprocessing
|
v
Model Training
|
v
Cross Validation
|
v
Final Model
|
v
Kaggle Prediction

## Environment

Python >= 3.10

Tested on:

Python 3.12

## Current Performance

### Validation Performance

- Best Model: **CatBoostClassifier**
- Validation Accuracy: **82.1%**
- 10-Fold Cross Validation Accuracy: **82.6%**

### Kaggle Performance

- Public Score: **0.77990**

The difference between validation and Kaggle performance is being investigated through further feature engineering and model improvements.


# Feature Engineering

The preprocessing pipeline creates additional features to better represent passenger characteristics.


## Passenger Features

### Title Extraction

Passenger titles are extracted from names.

Examples:

- Mr
- Mrs
- Miss
- Master
- Officer
- Royalty

Rare titles are grouped together to reduce noise.


### Age Imputation

Missing ages are filled using:

1. Median age grouped by passenger title
2. Overall median age as fallback


### Age Band

Passengers are grouped into age categories to capture nonlinear relationships between age and survival.


## Family Features

### Family Size

Calculated using:

```text
FamilySize = SibSp + Parch + 1
```

### Family Size Group

Passengers are categorised into:

- Alone
- Small Family
- Medium Family
- Large Family


### Is Alone

Binary feature:

```text
1 = Passenger travelled alone
0 = Passenger travelled with family
```


### WomanChild

Captures the historical "women and children first" survival pattern.

```text
Female OR Age < 16
```


## Ticket Features

### Ticket Group Size

Counts passengers sharing the same ticket.

Only training data is used to prevent test distribution leakage.


## Cabin Features

### Deck

Extracts the cabin deck letter.

Missing cabin values are grouped as:

```text
U = Unknown
```


### CabinKnown

Indicates whether cabin information was available.

```text
1 = Cabin available
0 = Cabin missing
```


## Fare Features

### Fare Per Person

Calculated as:

```text
Fare / FamilySize
```


### Fare Band

Passengers are grouped into fare categories.


# Models Tested

The following machine learning models were evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Tree ensemble model |
| Extra Trees | Randomised tree ensemble |
| XGBoost | Gradient boosting model |
| LightGBM | Gradient boosting model |
| CatBoost | Categorical boosting model |


# Final Model

The current best-performing model is:

```text
CatBoostClassifier
```

CatBoost was selected because:

- It performs well on small tabular datasets
- It handles categorical features effectively
- It achieved the highest validation consistency


The final model is retrained using the complete training dataset before generating Kaggle predictions.


# Project Structure

```
titanic-kaggle/

├── data/
│   ├── train.csv
│   └── test.csv
│
├── outputs/
│   ├── titanic_model.pkl
│   ├── model_columns.pkl
│   └── submission.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── README.md
└── experiments.md
```


# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```


## Train Model

```bash
python src/train.py
```


## Generate Submission

```bash
python src/predict.py
```


The generated submission file will be:

```text
outputs/submission.csv
```


# Future Improvements

Potential improvements:

- Family survival features
- Better ensemble methods
- Stacking classifiers
- SHAP feature importance analysis
- More extensive hyperparameter optimisation
- Error analysis on misclassified passengers


# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Kaggle Titanic Dataset
