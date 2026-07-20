# Titanic Survival Prediction - Machine Learning Learning Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Learning%20Project-green)

# ⚠️ Learning Project Disclaimer

This repository is a **learning project** created as my first machine learning test project.

I am currently a **beginner/student learning data science and machine learning**, and this project was built to understand the complete machine learning workflow:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Hyperparameter tuning
* Prediction generation

This project is not intended to represent an industry-level machine learning solution. It is mainly a personal learning exercise to explore how machine learning models work.

A significant portion of this project was developed with the assistance of **AI tools**. AI was used heavily for:

* Understanding machine learning concepts
* Debugging errors
* Suggesting code improvements
* Explaining model behaviour
* Exploring feature engineering approaches

The goal of this project was not only to create a prediction model, but also to learn how machine learning projects are structured and developed.

---

# Titanic: Machine Learning from Disaster

This project is based on the famous Kaggle competition:

**Titanic - Machine Learning from Disaster**

The objective is to predict whether a passenger survived the Titanic disaster using information such as:

* Passenger class
* Gender
* Age
* Fare
* Embarkation location
* Passenger title
* Family/group information
* Cabin availability

The project uses supervised machine learning classification models to predict the `Survived` target variable.

---

# Project Workflow

The project follows a complete machine learning pipeline:

```
Raw Dataset
     |
     v
Data Cleaning
     |
     v
Feature Engineering
     |
     v
Feature Encoding
     |
     v
Model Training
     |
     v
Cross Validation
     |
     v
Model Selection
     |
     v
Prediction
     |
     v
Submission File
```

---

# Features Used

The final feature set was selected after testing multiple feature combinations.

Current features:

| Feature         | Description                                             |
| --------------- | ------------------------------------------------------- |
| Pclass          | Passenger ticket class                                  |
| Sex             | Passenger gender                                        |
| Age             | Passenger age                                           |
| Fare            | Ticket fare                                             |
| Embarked        | Boarding location                                       |
| Title           | Extracted passenger title (Mr, Mrs, Miss, Master, etc.) |
| FamilySizeGroup | Categorised family size                                 |
| TicketGroupSize | Number of passengers sharing the ticket                 |
| CabinKnown      | Whether cabin information exists                        |
| WomanChild      | Indicates women and children groups                     |

---

# Feature Engineering

The preprocessing pipeline includes:

## Missing Value Handling

* Age values are filled using grouped passenger information
* Missing embarked values are handled
* Missing cabin values are converted into a binary feature

## Title Extraction

Passenger names are converted into titles:

Examples:

```
Mr
Mrs
Miss
Master
Officer
Professional
Royalty
```

## Family Features

Created features to capture survival patterns:

* Family size categories
* Ticket group size
* Woman/child indicator

---

# Models Tested

The project compares multiple classification algorithms:

## Logistic Regression

A baseline linear classification model.

## Random Forest

An ensemble tree model that combines multiple decision trees.

## Extra Trees

A randomized tree ensemble model.

## XGBoost

A gradient boosting algorithm designed for high-performance tabular data.

## CatBoost

A gradient boosting algorithm that performs well with categorical features.

---

# Model Evaluation

Models are evaluated using:

* Stratified K-Fold Cross Validation
* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

Latest cross-validation results:

```
LogisticRegression : 0.8193
RandomForest       : 0.8339
ExtraTrees         : 0.8316
XGBoost            : 0.8294
CatBoost            : 0.8384
```

Best performing model:

```
CatBoostClassifier
```

---

# Project Structure

```
titanic-kaggle/

│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── outputs/
│   ├── titanic_model.pkl
│   ├── model_columns.pkl
│   ├── threshold.pkl
│   └── submission.csv
│
├── assets/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
└── README.md
```

---

# How To Run

## Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn joblib
```

## Train the models

```bash
python src/train.py
```

This will:

* Preprocess the dataset
* Train multiple models
* Perform cross-validation
* Select the best model
* Save the trained model

---

## Generate Predictions

```bash
python src/predict.py
```

This creates:

```
outputs/submission.csv
```

which can be submitted to Kaggle.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Matplotlib
* Seaborn
* Joblib

---

# What I Learned

Through this project, I learned:

* How machine learning pipelines are structured
* The importance of preprocessing
* How feature engineering affects model performance
* Differences between machine learning algorithms
* How to evaluate classification models
* How to prepare Kaggle submissions

This project is my first step into machine learning and data science, and I will continue improving my understanding through future projects.

---

# Future Improvements

Possible improvements for future learning:

* Better hyperparameter optimisation
* Ensemble stacking/voting methods
* More advanced feature engineering
* Better validation strategies
* Experiment tracking
* Model deployment

---

## Final Note

This repository represents my learning journey into machine learning. The project was built with significant AI assistance as a learning tool, while I focused on understanding the workflow, experimenting with models, and learning from the results.
