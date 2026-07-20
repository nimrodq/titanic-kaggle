# Titanic Kaggle Experiments

## Project Overview

This project builds a complete machine learning pipeline for the Kaggle Titanic competition.

The workflow includes:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Cross-validation
- Kaggle submission generation
- Experiment tracking using Git

---

# Experiment Results

| Experiment | Model | Changes | Public Kaggle Score |
|------------|-------|---------|--------------------:|
| Baseline | Logistic Regression | Pclass, Sex, Age, Fare, FamilySize, IsAlone, Embarked, Title | **0.77751** |
| Feature Engineering | Logistic Regression | Added Rare Title handling + FarePerPerson | 0.77511 |
| Model Comparison | Random Forest | Same baseline features | 0.77751 |
| Feature Engineering | Logistic Regression | Improved Age imputation | 0.77511 |
| Feature Engineering | Logistic Regression | Added Cabin Deck feature | 0.76794 |
| Model Comparison | Gradient Boosting | Age bands, Fare bands, TicketGroup, Deck and additional engineered features | 0.76555 |
| Hyperparameter Tuning | Logistic Regression | C=0.5, solver=liblinear | **0.77990** ✅ |
| Feature Engineering | Logistic Regression | Added TicketGroup + HasCabin | 0.77751 |
| Ensemble Learning | Soft Voting Classifier | Logistic Regression + Random Forest + Gradient Boosting with 5-Fold Stratified Cross-Validation | 0.77751 |

---

# Best Model

## Logistic Regression

### Parameters

```
C = 0.5
solver = liblinear
max_iter = 1000
```

### Features

- Pclass
- Sex
- Age
- Fare
- FamilySize
- IsAlone
- Embarked
- Title

### Best Public Kaggle Score

**0.77990**

---

# Cross-Validation Experiment

To evaluate a more robust machine learning workflow, a Soft Voting Ensemble was tested.

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting

## Validation Method

- Stratified 5-Fold Cross-Validation

## Cross-Validation Scores

```
Fold 1 : 0.83799
Fold 2 : 0.81461
Fold 3 : 0.83146
Fold 4 : 0.83146
Fold 5 : 0.84831
```

**Mean Cross-Validation Accuracy: 0.8328**

Although the ensemble achieved a higher cross-validation accuracy than the Logistic Regression model, its Kaggle public leaderboard score (0.77751) was lower than the tuned Logistic Regression model (0.77990). This demonstrates that better validation performance does not always translate into better generalization on unseen test data.

---

# Feature Engineering Tested

## Title Extraction

Extracted passenger titles from names.

Grouped uncommon titles into a single **Rare** category.

Examples:

- Mr
- Mrs
- Miss
- Master
- Rare

Rare titles included:

- Dr
- Rev
- Lady
- Major
- Sir
- Countess
- Capt
- Col
- Don
- Jonkheer

---

## Family Size

Created:

```
FamilySize = SibSp + Parch + 1
```

This captures the total travelling group size.

---

## IsAlone

Created:

```
IsAlone = (FamilySize == 1)
```

Passengers travelling alone exhibited different survival patterns.

---

## Ticket Group

Created:

```
TicketGroup = Number of passengers sharing the same ticket
```

Did not improve leaderboard performance.

---

## Cabin Features

Tested:

- Deck
- HasCabin

Neither feature improved the final Kaggle score.

---

# Models Evaluated

## Logistic Regression

Advantages:

- Strong baseline for small datasets
- Fast to train
- Less prone to overfitting
- Best leaderboard performance

Public Score:

**0.77990**

---

## Random Forest

Advantages:

- Captures nonlinear relationships
- Handles feature interactions

Public Score:

**0.77751**

---

## Gradient Boosting

Advantages:

- Powerful nonlinear learner

Result:

Performed worse than Logistic Regression on the hidden Kaggle test set.

Public Score:

**0.76555**

---

## Soft Voting Ensemble

Components:

- Logistic Regression
- Random Forest
- Gradient Boosting

Validation:

- 5-Fold Stratified Cross-Validation

Mean CV Accuracy:

**0.8328**

Public Kaggle Score:

**0.77751**

The ensemble achieved the highest validation accuracy but did not outperform the tuned Logistic Regression model on Kaggle, indicating slight overfitting to the training data.

---

# Lessons Learned

This project highlighted several important machine learning principles:

- Simpler models can outperform more complex models on small datasets.
- Feature engineering should be validated experimentally rather than assumed to improve performance.
- Hyperparameter tuning provided the largest improvement (+0.00239 leaderboard score).
- Cross-validation offers a more reliable estimate of model performance, but higher validation accuracy does not always lead to a higher public leaderboard score.
- Reproducible experiments and version control made it easy to compare different approaches and restore the best-performing model.

---

# Final Conclusion

The final selected model is:

**Logistic Regression**

```
C = 0.5
solver = liblinear
max_iter = 1000
```

Final Public Kaggle Score:

# **0.77990**

While more sophisticated approaches such as Gradient Boosting and Soft Voting Ensembles were explored, the tuned Logistic Regression model consistently provided the best balance between simplicity, generalization, and leaderboard performance.