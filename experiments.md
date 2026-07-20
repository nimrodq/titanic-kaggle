# Titanic Model Experiments

This document records the machine learning experiments conducted while developing my Titanic survival prediction model.

## Learning Project Disclaimer

This is my **first machine learning test project** as a beginner/student learning data science.

The purpose of this project was to understand the complete machine learning workflow:

* Data preprocessing
* Feature engineering
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Prediction generation

This project was developed with significant assistance from **AI tools**, which were used for:

* Understanding machine learning concepts
* Debugging errors
* Suggesting improvements
* Explaining model behaviour
* Exploring feature engineering ideas

The main goal was learning and experimentation rather than creating a production-level model.

---

# Saved Artifacts

## `titanic_model.pkl`

Contains the final trained machine learning model selected from experiments.

Current best model:

```text
CatBoostClassifier
```

---

## `model_columns.pkl`

Stores the final feature column order after encoding.

This ensures that training and prediction data use the same feature structure.

---

## `threshold.pkl`

Stores the classification threshold used during prediction.

Current threshold:

```text
0.45
```

---

# Experiment 1 - Baseline Model

## Objective

Create a simple baseline model using only basic passenger information.

## Model

```text
Logistic Regression
```

## Features

Initial features:

* Pclass
* Sex
* Age
* Fare
* Embarked

## Result

Validation accuracy:

```text
~78%
```

The baseline model provided a starting point but was unable to capture more complex passenger survival patterns.

---

# Experiment 2 - Feature Engineering

## Objective

Improve prediction performance by creating more meaningful passenger features.

## Added Features

## Passenger Information

Added:

* Passenger Title extraction
* Age handling based on passenger information

Examples of extracted titles:

```text
Mr
Mrs
Miss
Master
Officer
Professional
Royalty
```

---

## Family Features

Added:

* FamilySizeGroup
* TicketGroupSize

These features help capture relationships between passengers travelling together.

---

## Cabin Features

Added:

* CabinKnown

This identifies whether cabin information exists.

---

## Survival Pattern Feature

Added:

* WomanChild

This feature combines gender and age information based on the historical evacuation priority:

```text
Women and children first
```

---

## Removed Features

During experimentation, some features were removed because they added noise or redundancy.

Removed:

* FamilySize
* IsAlone
* FareBand
* SexClass
* TitleClass
* ChildClass

Reason:

These features either duplicated existing information or increased feature complexity without improving validation performance.

---

# Experiment 3 - Model Comparison

## Objective

Compare different machine learning algorithms on the engineered dataset.

## Models Tested

---

# Logistic Regression

Advantages:

* Simple baseline model
* Interpretable coefficients
* Works well with encoded categorical variables

Result:

```text
~81.9% Cross Validation Accuracy
```

---

# Random Forest

Advantages:

* Handles nonlinear relationships
* Robust ensemble method

Result:

```text
~83.4% Cross Validation Accuracy
```

---

# Extra Trees

Advantages:

* Similar to Random Forest
* More randomised tree construction

Result:

```text
~83.2% Cross Validation Accuracy
```

---

# XGBoost

Advantages:

* Gradient boosting algorithm
* Strong performance on structured datasets

Result:

```text
~82.9% Cross Validation Accuracy
```

---

# CatBoost

Advantages:

* Effective gradient boosting algorithm
* Performs well on small tabular datasets

Result:

```text
~83.8% Cross Validation Accuracy
```

---

# Experiment 4 - Hyperparameter Tuning

## Objective

Improve model performance by tuning important model parameters.

## Tuned Models

---

## Random Forest

Parameters adjusted:

* Number of estimators
* Maximum depth
* Minimum samples split
* Minimum samples leaf

Final settings:

```text
n_estimators = 1000
max_depth = 6
```

---

## XGBoost

Parameters adjusted:

* Number of estimators
* Learning rate
* Maximum depth
* Subsampling
* Regularisation

---

## CatBoost

Parameters adjusted:

* Iterations
* Depth
* Learning rate
* L2 regularisation

Final settings:

```text
iterations = 1000
depth = 4
learning_rate = 0.02
l2_leaf_reg = 5
```

---

# Experiment 5 - Stratified K-Fold Cross Validation

## Objective

Reduce the impact of a lucky or unlucky validation split.

## Method

Used:

```text
5-Fold Stratified Cross Validation
```

Stratification ensures that each fold maintains a similar survival class distribution.

---

## Result

Final cross-validation results:

| Model               | Mean Accuracy |
| ------------------- | ------------: |
| Logistic Regression |        81.93% |
| Random Forest       |        83.39% |
| Extra Trees         |        83.16% |
| XGBoost             |        82.94% |
| CatBoost            |        83.84% |

---

# Experiment 6 - Feature Selection

## Objective

Reduce unnecessary complexity and improve generalisation.

## Final Feature Set

The final model uses:

| Feature         |
| --------------- |
| Pclass          |
| Sex             |
| Age             |
| Fare            |
| Embarked        |
| Title           |
| FamilySizeGroup |
| TicketGroupSize |
| CabinKnown      |
| WomanChild      |

---

## Feature Encoding

Categorical features were converted using:

```python
pd.get_dummies(
    drop_first=True
)
```

This avoids the dummy variable trap by removing redundant categories.

Example:

Before:

```text
Sex_female
Sex_male
```

After:

```text
Sex_male
```

where:

```text
1 = Male
0 = Female
```

---

# Current Best Model

Final selected model:

```text
CatBoostClassifier
```

## Cross Validation Performance

| Metric                |  Score |
| --------------------- | -----: |
| Mean CV Accuracy      | 83.84% |
| CV Standard Deviation |  1.49% |

---

## Final Feature Importance

Most influential features:

| Feature         | Importance |
| --------------- | ---------: |
| Fare            |       High |
| Age             |       High |
| Title_Mr        |       High |
| WomanChild      |       High |
| Pclass          |       High |
| TicketGroupSize |     Medium |
| FamilySizeGroup |     Medium |
| Sex_male        |     Medium |

The results show that passenger characteristics such as age, gender, class, and economic status strongly influenced survival prediction.

---

# Experiment 7 - Error Analysis

## Objective

Understand where the model makes incorrect predictions.

## Observations

The model struggles mainly with:

### False Negatives

Passengers predicted not to survive but actually survived.

Common patterns:

* Male passengers
* Lower-class passengers
* Passengers with limited information

---

### False Positives

Passengers predicted to survive but did not.

Common patterns:

* Female passengers
* Young passengers
* Passengers without strong family/ticket indicators

---

# Future Learning Improvements

Possible future experiments:

## Ensemble Methods

Explore:

* Voting Classifier
* Stacking Classifier

## Explainability

Possible additions:

* SHAP analysis
* More detailed error analysis
* Feature contribution visualisation

## Advanced Validation

Future improvements:

* More robust experiment tracking
* Hyperparameter optimisation tools
* Automated model comparison

---

# Final Reflection

This project represents my first step into machine learning and data science.

Through this experiment, I learned:

* How machine learning pipelines are structured
* How preprocessing affects model performance
* How feature engineering changes predictions
* How different algorithms behave
* How to evaluate and compare models

Although AI tools were heavily used throughout development, the purpose was to use them as a learning assistant while building my understanding of machine learning concepts. This is my first real repository and first real learning experience. Thank you for reading.
