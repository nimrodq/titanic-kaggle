# Titanic Kaggle Experiments

## Project Overview

This project builds a machine learning pipeline for the Kaggle Titanic competition.

The workflow includes:

* Data preprocessing
* Feature engineering
* Model training
* Validation testing
* Kaggle submission generation

---

# Experiment Results

| Experiment            | Model               | Features / Changes                                           | Kaggle Public Score |
| --------------------- | ------------------- | ------------------------------------------------------------ | ------------------: |
| Baseline              | Logistic Regression | Pclass, Sex, Age, Fare, FamilySize, IsAlone, Embarked, Title |             0.77751 |
| Feature Experiment    | Logistic Regression | Added Rare Title handling + FarePerPerson                    |             0.77511 |
| Model Experiment      | Random Forest       | Same baseline features                                       |             0.77751 |
| Feature Experiment    | Logistic Regression | Age imputation improvements                                  |             0.77511 |
| Feature Experiment    | Logistic Regression | Added Cabin Deck feature                                     |             0.76794 |
| Model Experiment      | Gradient Boosting   | Age bands, Fare bands, TicketGroup, Deck, extra features     |             0.76555 |
| Hyperparameter Tuning | Logistic Regression | C=0.5, solver=liblinear                                      |         **0.77990** |
| Feature Experiment    | Logistic Regression | Added TicketGroup + HasCabin                                 |             0.77751 |

---

# Best Model

## Logistic Regression

Parameters:

```text
C = 0.5
solver = liblinear
max_iter = 1000
```

Features:

```text
Pclass
Sex
Age
Fare
FamilySize
IsAlone
Embarked
Title
```

Best Kaggle Public Score:

```text
0.77990
```

---

# Feature Engineering Tested

## Title Extraction

Extracted titles from passenger names:

Examples:

```text
Mr
Mrs
Miss
Master
Rare
```

Rare titles were grouped together:

```text
Dr
Rev
Sir
Lady
Major
```

---

## Family Size

Created:

```python
FamilySize = SibSp + Parch + 1
```

This captures the size of a passenger's travelling group.

---

## Is Alone

Created:

```python
IsAlone = FamilySize == 1
```

Passengers travelling alone had different survival patterns.

---

## Ticket Group Size

Tested:

```python
TicketGroup
```

Number of passengers sharing the same ticket.

This did not improve the final leaderboard score.

---

## Cabin Features

Tested:

```python
Deck
HasCabin
```

These features reduced leaderboard performance compared to the tuned baseline.

---

# Models Tested

## Logistic Regression

Best performing model.

Advantages:

* Works well on small datasets
* Less prone to overfitting
* Fast training

---

## Random Forest

Result:

```text
0.77751
```

Did not outperform Logistic Regression.

---

## Gradient Boosting

Result:

```text
0.76555
```

The additional complexity did not generalize better.

---

# Final Conclusion

The best-performing approach was a simple Logistic Regression model with tuned regularization.

Additional feature engineering and more complex models did not improve leaderboard performance.

Final selected model:

```text
Logistic Regression
C=0.5
solver=liblinear
```

Final Kaggle Public Score:

```text
0.77990
```
