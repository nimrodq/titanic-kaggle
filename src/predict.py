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



train, test = preprocess_data(train, test)

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

X_test = pd.get_dummies(X_test)

X_test = X_test.fillna(
    X_test.median(numeric_only=True)
)

model = joblib.load(
    "outputs/titanic_model.pkl"
)

model_columns = joblib.load(
    "outputs/model_columns.pkl"
)

X_test = X_test.reindex(
    columns=model_columns,
    fill_value=0
)

predictions = model.predict(X_test)

submission = pd.DataFrame({

    "PassengerId": test_ids,

    "Survived": predictions

})

print(X_test.head())
print(X_test.shape)
print(model.feature_names_)
print(submission["Survived"].value_counts())
print(type(model))

submission.to_csv(
    "outputs/submission.csv",
    index=False
)

print("Submission created!")