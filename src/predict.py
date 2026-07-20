import pandas as pd
import numpy as np
import joblib

from preprocess import preprocess_data


# ==========================
# Load Data
# ==========================

train = pd.read_csv(
    "data/train.csv"
)

test = pd.read_csv(
    "data/test.csv"
)


train, test = preprocess_data(
    train,
    test
)


# ==========================
# Save Passenger IDs
# ==========================

test_ids = test["PassengerId"]



# ==========================
# Prepare Features
# ==========================

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "FamilyGroup",
    "FarePerPerson",
    "TicketPrefix"
]


X_test = test[features]


# Encode categorical variables

X_test = pd.get_dummies(
    X_test
)


# Fill missing values

X_test = X_test.fillna(
    X_test.median(numeric_only=True)
)



# ==========================
# Load Model
# ==========================

model = joblib.load(
    "outputs/titanic_model.pkl"
)


model_columns = joblib.load(
    "outputs/model_columns.pkl"
)



# Match training columns

X_test = X_test.reindex(
    columns=model_columns,
    fill_value=0
)



# ==========================
# Predict
# ==========================

predictions = model.predict(
    X_test
)



# ==========================
# Submission
# ==========================

submission = pd.DataFrame({

    "PassengerId": test_ids,

    "Survived": predictions

})



# Debug

print(X_test.head())

print(
    X_test.shape
)

print(
    model.feature_names_
)

print(
    submission["Survived"].value_counts()
)

print(
    type(model)
)



# ==========================
# Save Submission
# ==========================

submission.to_csv(
    "outputs/submission.csv",
    index=False
)


print(
    "Submission created!"
)