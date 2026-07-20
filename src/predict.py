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
# Prepare Test
# ==========================

test_ids = test["PassengerId"]



X_test = test.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]
)



# ==========================
# Load Encoders
# ==========================

encoders = joblib.load(
    "encoders.pkl"
)



for col, encoder in encoders.items():

    X_test[col] = encoder.transform(
        X_test[col].astype(str)
    )



# ==========================
# Clean Data
# ==========================

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


X_test = X_test.fillna(
    0
)



# ==========================
# Load Model
# ==========================

model = joblib.load(
    "model.pkl"
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

print(X_test.head())
print(X_test.shape)
print(model.feature_names_)
print(submission["Survived"].value_counts())
print(type(model))

submission.to_csv(
    "outputs/submission.csv",
    index=False
)



print(
    "submission.csv created!"
)