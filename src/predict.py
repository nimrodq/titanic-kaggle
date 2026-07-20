import pandas as pd
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
# Features
# ==========================

features = [

    "Pclass",

    "Sex",

    "Age",

    "Fare",

    "Embarked",

    "Title",

    "FamilySizeGroup",

    "TicketGroupSize",

    "CabinKnown",

    "WomanChild"

]


X_test = test[features]



# ==========================
# Encoding
# ==========================

X_test = pd.get_dummies(
    X_test,
    drop_first=True
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


threshold = joblib.load(
    "outputs/threshold.pkl"
)



# ==========================
# Match Training Columns
# ==========================

X_test = X_test.reindex(

    columns=model_columns,

    fill_value=0

)



# ==========================
# Prediction
# ==========================

probabilities = model.predict_proba(
    X_test
)[:,1]


predictions = (

    probabilities > threshold

).astype(int)



# ==========================
# Create Submission
# ==========================

submission = pd.DataFrame({

    "PassengerId": test_ids,

    "Survived": predictions

})



# ==========================
# Debug
# ==========================

print(
    X_test.head()
)

print(
    "Shape:",
    X_test.shape
)

print(
    "Model:",
    type(model)
)

print(
    "Threshold:",
    threshold
)

print(
    "Prediction Distribution:"
)

print(
    submission["Survived"].value_counts()
)



# ==========================
# Save Submission
# ==========================

submission.to_csv(

    "outputs/submission.csv",

    index=False

)


print(
    "\nSubmission created!"
)